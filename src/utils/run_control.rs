use crate::utils::{ActiveRunRecord, ChatHistory, ChatMode, RunStatus};
use dioxus::core::Task;
use dioxus::prelude::{ReadableExt, Signal, WritableExt};
use futures::Stream;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::mpsc;

use super::{ModelStreamEvent, StreamEvent};

pub fn create_run_id(mode: ChatMode, session_id: &Option<String>) -> String {
    let session_part = session_id
        .clone()
        .unwrap_or_else(|| "ephemeral".to_string());
    format!(
        "{:?}-{}-{}",
        mode,
        session_part,
        ChatHistory::format_timestamp()
    )
}

pub fn register_active_run(
    mut active_runs: Signal<HashMap<String, ActiveRunRecord>>,
    id: String,
    session_id: Option<String>,
    mode: ChatMode,
    label: String,
    task: Task,
    cancel_flag: Arc<AtomicBool>,
) {
    active_runs.write().insert(
        id.clone(),
        ActiveRunRecord {
            id,
            session_id,
            mode,
            label,
            status: RunStatus::Running,
            started_at: ChatHistory::format_timestamp(),
            task,
            cancel_flag,
        },
    );
}

pub fn set_run_status(
    mut active_runs: Signal<HashMap<String, ActiveRunRecord>>,
    run_id: &str,
    status: RunStatus,
) {
    if let Some(run) = active_runs.write().get_mut(run_id) {
        run.status = status;
    }
}

pub fn remove_run(mut active_runs: Signal<HashMap<String, ActiveRunRecord>>, run_id: &str) {
    active_runs.write().remove(run_id);
}

pub fn find_run_for_session(
    active_runs: Signal<HashMap<String, ActiveRunRecord>>,
    session_id: &Option<String>,
    mode: ChatMode,
) -> Option<ActiveRunRecord> {
    active_runs
        .read()
        .values()
        .find(|run| run.session_id == *session_id && run.mode == mode)
        .cloned()
}

pub fn is_cancelled(cancel_flag: &Arc<AtomicBool>) -> bool {
    cancel_flag.load(Ordering::SeqCst)
}

pub fn try_signal_set<T: 'static>(signal: &mut Signal<T>, value: T) -> bool {
    match signal.try_write() {
        Ok(mut write) => {
            *write = value;
            true
        }
        Err(_) => false,
    }
}

pub fn try_signal_update<T: 'static, R>(
    signal: &mut Signal<T>,
    f: impl FnOnce(&mut T) -> R,
) -> Option<R> {
    match signal.try_write() {
        Ok(mut write) => Some(f(&mut *write)),
        Err(_) => None,
    }
}

pub fn try_signal_read<T: 'static, R>(
    signal: &Signal<T>,
    f: impl FnOnce(&T) -> R,
) -> Option<R> {
    match signal.try_read() {
        Ok(read) => Some(f(&*read)),
        Err(_) => None,
    }
}

pub async fn next_stream_event_with_cancel<S>(
    stream: &mut S,
    cancel_flag: &Arc<AtomicBool>,
) -> Option<StreamEvent>
where
    S: Stream<Item = StreamEvent> + Unpin,
{
    loop {
        if is_cancelled(cancel_flag) {
            return Some(StreamEvent::Error("Cancelled".to_string()));
        }

        tokio::select! {
            biased;
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                if is_cancelled(cancel_flag) {
                    return Some(StreamEvent::Error("Cancelled".to_string()));
                }
            }
            next = stream.next() => return next,
        }
    }
}

pub async fn recv_multi_event_with_cancel(
    rx: &mut mpsc::UnboundedReceiver<ModelStreamEvent>,
    cancel_flag: &Arc<AtomicBool>,
) -> Option<ModelStreamEvent> {
    loop {
        if is_cancelled(cancel_flag) {
            return Some(ModelStreamEvent {
                model_id: "__cancelled__".to_string(),
                event: StreamEvent::Error("Cancelled".to_string()),
            });
        }

        tokio::select! {
            biased;
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                if is_cancelled(cancel_flag) {
                    return Some(ModelStreamEvent {
                        model_id: "__cancelled__".to_string(),
                        event: StreamEvent::Error("Cancelled".to_string()),
                    });
                }
            }
            next = rx.recv() => return next,
        }
    }
}
