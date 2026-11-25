//! Explicitly selected, small hardware checks for the lean-window path.
//! No game data, shader compilation or shader dispatch is involved.

use super::*;
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::time::Duration;
use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};

struct Event(HANDLE);

impl Event {
    fn new() -> Self {
        // SAFETY: a new auto-reset event, owned by this helper.
        Self(unsafe { CreateEventW(None, false, false, None).expect("test event") })
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: owned here, and every registered wait finished first.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

fn check_device(host: &Host) {
    // SAFETY: a plain query on our live device.
    unsafe {
        host.device
            .GetDeviceRemovedReason()
            .expect("test device removed")
    };
    assert!(host.log_messages().is_none(), "D3D12 validation error");
}

fn fence(host: &Host) -> ID3D12Fence {
    // SAFETY: plain creation on the live device.
    unsafe {
        host.device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .expect("test fence")
    }
}

fn wait_for(host: &Host, fence: &ID3D12Fence, value: u64) {
    check_device(host);
    let event = Event::new();
    // SAFETY: our fence and event; the event lives through the wait.
    unsafe {
        fence
            .SetEventOnCompletion(value, event.0)
            .expect("test fence event");
        if WaitForSingleObject(event.0, 5_000) != WAIT_OBJECT_0 {
            // Do not unwind and release resources the queue may still use.
            eprintln!("lean GPU test fence did not complete within 5 seconds");
            std::process::abort();
        }
    }
    check_device(host);
}

fn readback(host: &Host, size: u64) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut resource = None;
    // SAFETY: a small CPU-readable destination for these tests only.
    unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut resource,
            )
            .expect("test readback buffer");
    }
    resource.expect("test readback resource")
}

/// Only called after the fence behind the copy into this resource completed.
fn read_bytes(resource: &ID3D12Resource, size: usize) -> Vec<u8> {
    let mut data = std::ptr::null_mut();
    let range = D3D12_RANGE {
        Begin: 0,
        End: size,
    };
    // SAFETY: the resource is a readback buffer of `size` bytes, its GPU
    // copy is done, and the mapping is kept until the bytes are copied.
    unsafe {
        resource
            .Map(0, Some(&range), Some(&mut data))
            .expect("map test readback");
        let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size).to_vec();
        resource.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        bytes
    }
}

fn assert_bytes(actual: &[u8], expected: &[u8], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: readback length");
    if let Some(at) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{what}: first mismatch at {at:#x}, got {:#04x}, expected {:#04x}",
            actual[at], expected[at]
        );
    }
}

fn record(host: &Host, commands: impl FnOnce(&ID3D12GraphicsCommandList)) -> Prelude {
    let (allocator, list) = host.make_prelude().expect("test command list");
    // SAFETY: this new allocator and closed list have never run.
    unsafe {
        allocator.Reset().expect("reset test allocator");
        list.Reset(&allocator, None).expect("reset test list");
    }
    commands(&list);
    // SAFETY: the list is open and recording is done.
    unsafe { list.Close().expect("close test list") };
    (allocator, list)
}

fn execute_and_wait(host: &Host, prelude: &Prelude) {
    let done = fence(host);
    // SAFETY: closed list; all resources its commands name are retained
    // by the caller until this fence has completed.
    unsafe {
        host.queue
            .ExecuteCommandLists(&[Some(prelude.1.cast().expect("test list interface"))]);
        if let Err(error) = host.queue.Signal(&done, 1) {
            eprintln!("lean GPU test could not signal its fence: {error}");
            std::process::abort();
        }
    }
    wait_for(host, &done, 1);
}

#[test]
#[ignore = "small hardware GPU test; run this exact test in its own process"]
fn recycled_tiles_are_cleared_before_partial_uploads() {
    let host = host().expect("D3D12 device required for lean GPU test");
    let _submission = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    check_device(host);
    const SIZE: u64 = 4 * TILE;
    let Some(a) = host.reserved_buffer(SIZE) else {
        eprintln!(
            "SKIP recycled tile test: reserved buffers / tiled resources tier 2 are unavailable"
        );
        return;
    };
    let tiles = host.take_tiles(2).expect("two test pool tiles");
    let maps_a = [(0, tiles[0]), (2, tiles[1])];
    host.map_tiles(&a, &maps_a);
    let (source, source_data) = host.upload_buffer(2 * TILE).expect("test pattern upload");
    // SAFETY: two writable tiles of our upload resource.
    unsafe {
        source_data.write_bytes(0xa5, TILE as usize);
        source_data
            .add(TILE as usize)
            .write_bytes(0x3c, TILE as usize);
    }
    let first = readback(host, SIZE);
    let initial = record(host, |list| {
        // SAFETY: mapped tiles 0/2 and buffers large enough for every copy.
        unsafe {
            list.CopyBufferRegion(&a, 0, &source, 0, TILE);
            list.CopyBufferRegion(&a, 2 * TILE, &source, TILE, TILE);
            list.ResourceBarrier(&[transition(
                &a,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyBufferRegion(&first, 0, &a, 0, SIZE);
        }
    });
    execute_and_wait(host, &initial);
    let initial_bytes = read_bytes(&first, SIZE as usize);
    assert!(initial_bytes[..TILE as usize].iter().all(|&b| b == 0xa5));
    assert!(
        initial_bytes[TILE as usize..(2 * TILE) as usize]
            .iter()
            .all(|&b| b == 0)
    );
    assert!(
        initial_bytes[(2 * TILE) as usize..(3 * TILE) as usize]
            .iter()
            .all(|&b| b == 0x3c)
    );
    assert!(initial_bytes[(3 * TILE) as usize..].iter().all(|&b| b == 0));

    // The fence completed: the old resource can go and its tiles can be
    // reassigned, just as a retired lean window returns them to the pool.
    drop(a);
    give_back_tiles(tiles.clone());
    let recycled = host.take_tiles(2).expect("recycled test tiles");
    assert_eq!(
        recycled, tiles,
        "the same physical pool tiles must be reused"
    );
    let b = host.reserved_buffer(SIZE).expect("second reserved buffer");
    let maps_b = [(1, recycled[0]), (3, recycled[1])];
    host.map_tiles(&b, &maps_b);
    let before = readback(host, SIZE);
    let inherited = record(host, |list| {
        // SAFETY: the new buffer's two tiles map the previous patterns.
        unsafe { list.CopyBufferRegion(&before, 0, &b, 0, SIZE) };
    });
    execute_and_wait(host, &inherited);
    let inherited_bytes = read_bytes(&before, SIZE as usize);
    if inherited_bytes[TILE as usize..(2 * TILE) as usize]
        .iter()
        .all(|&v| v == 0xa5)
        && inherited_bytes[(3 * TILE) as usize..]
            .iter()
            .all(|&v| v == 0x3c)
    {
        eprintln!(
            "CONTROL: both recycled tiles retained their nonzero bytes before initialization"
        );
    } else {
        eprintln!(
            "CONTROL INCONCLUSIVE: this adapter did not expose the old tile patterns; the positive zero/partial-upload check still runs"
        );
    }

    let (zero, seed) = host
        .make_zero_tile()
        .expect("production zero tile and upload seed");
    let (payload, payload_data) = host.upload_buffer(5).expect("test partial payload");
    let pattern = [0x11, 0x22, 0x33, 0x44, 0x55];
    // SAFETY: owned upload mappings of these lengths.
    unsafe {
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), payload_data, pattern.len());
    }
    let after = readback(host, SIZE);
    let cleared = record(host, |list| {
        initialize_zero_tile(list, &zero, &seed);
        clear_new_tiles(list, &b, &maps_b, &zero);
        // SAFETY: the production helper leaves COPY_DEST; the payload
        // covers five bytes inside mapped tile 1, then all bytes are read.
        unsafe {
            list.CopyBufferRegion(&b, TILE + 7, &payload, 0, 5);
            list.ResourceBarrier(&[transition(
                &b,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyBufferRegion(&after, 0, &b, 0, SIZE);
        }
    });
    execute_and_wait(host, &cleared);
    let mut expected = vec![0u8; SIZE as usize];
    expected[TILE as usize + 7..TILE as usize + 12].copy_from_slice(&pattern);
    assert_bytes(
        &read_bytes(&after, SIZE as usize),
        &expected,
        "cleared recycled window",
    );
    drop(b);
    give_back_tiles(recycled);
    check_device(host);
}

#[test]
#[ignore = "small hardware GPU test; run this exact test in its own process"]
fn wrapped_ring_waits_before_overwriting_in_flight_bytes() {
    let host = host().expect("D3D12 device required for lean GPU test");
    let _submission = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    check_device(host);
    const SIZE: u64 = 128 << 10;
    let (upload, data) = host.upload_buffer(SIZE).expect("small test ring");
    // SAFETY: the whole upload mapping is writable and not in use yet.
    unsafe { data.write_bytes(0x2a, SIZE as usize) };
    let first = readback(host, SIZE);
    let second = readback(host, SIZE);
    let first_copy = record(host, |list| {
        // SAFETY: both buffers hold SIZE bytes.
        unsafe { list.CopyBufferRegion(&first, 0, &upload, 0, SIZE) };
    });
    let gate = fence(host);
    let copied = fence(host);
    let worker_upload = upload.clone();
    let worker_fence = copied.clone();
    let address = data as usize;
    let (started_tx, started_rx) = channel();
    let (done_tx, done_rx) = channel();
    let first_commands = first_copy.1.cast().expect("test copy interface");

    // From this point until the CPU signals the gate there are no asserts
    // or early returns: even a worker failure must release the GPU queue.
    unsafe { host.queue.Wait(&gate, 1).expect("queue test gate") };
    // SAFETY: the closed copy waits behind the gate; resources stay alive.
    unsafe { host.queue.ExecuteCommandLists(&[Some(first_commands)]) };
    let queued_signal = unsafe { host.queue.Signal(&copied, 1) };
    let worker = std::thread::Builder::new()
        .name("lean-ring-test".into())
        .spawn(move || -> windows::core::Result<()> {
            // The HANDLE and Ring are created here, never sent across threads.
            let event = Event::new();
            let mut ring = Ring {
                buffer: worker_upload,
                data: address,
                size: SIZE,
                head: SIZE,
                busy: [(0, SIZE, 1)].into_iter().collect(),
                fence: worker_fence,
                event: event.0,
                value: 1,
            };
            let (at, n) = ring_take(ring.head, ring.size, SIZE);
            started_tx.send(()).expect("notify ring wait start");
            // SAFETY: this worker owns the ring, the submission lock excludes
            // other host work, and the root retains all referenced objects.
            unsafe {
                ring.clear(at, at + n)?;
                ((ring.data + at as usize) as *mut u8).write_bytes(0xb4, n as usize);
            }
            done_tx.send(()).expect("notify ring overwrite completion");
            Ok(())
        });
    let started = started_rx.recv_timeout(Duration::from_secs(5));
    let early = done_rx.recv_timeout(Duration::from_millis(50));
    // Always open the gate BEFORE inspecting any results or joining.
    let released = unsafe { gate.Signal(1) };
    if released.is_err() || queued_signal.is_err() {
        eprintln!(
            "lean ring test could not release/signal its queue: release={released:?}, queued={queued_signal:?}"
        );
        std::process::abort();
    }
    wait_for(host, &copied, 1);
    let worker = worker.expect("create ring worker after releasing the queue gate");
    if early.is_err()
        && done_rx.recv_timeout(Duration::from_secs(5)).is_err()
        && !worker.is_finished()
    {
        eprintln!("lean ring test worker did not finish after its fence completed");
        std::process::abort();
    }
    let worker_result = worker.join();
    assert!(started.is_ok(), "ring worker failed to start: {started:?}");
    assert!(
        matches!(early, Err(RecvTimeoutError::Timeout)),
        "overlapping ring bytes were writable while the GPU was gated: {early:?}"
    );
    worker_result
        .expect("ring worker panicked")
        .expect("ring clear failed");
    assert_bytes(
        &read_bytes(&first, SIZE as usize),
        &vec![0x2a; SIZE as usize],
        "copy before ring overwrite",
    );

    let second_copy = record(host, |list| {
        // SAFETY: the worker is done; every byte now holds pattern B.
        unsafe { list.CopyBufferRegion(&second, 0, &upload, 0, SIZE) };
    });
    execute_and_wait(host, &second_copy);
    assert_bytes(
        &read_bytes(&second, SIZE as usize),
        &vec![0xb4; SIZE as usize],
        "copy after ring overwrite",
    );
    check_device(host);
}
