use super::common::{ChatInput, FormattedText, Modal, ModelSelector};
use crate::utils::{
    create_run_id, find_run_for_session, next_stream_event_with_cancel, register_active_run,
    remove_run, set_run_status, try_signal_read, try_signal_set, try_signal_update,
    ActiveRunRecord, ChatMessage, ChatHistory, ChatMode, ChatSession, InputSettings,
    OpenRouterClient, RunStatus, SessionData, StandardHistory, StreamEvent, Theme,
};
use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone, Debug, PartialEq)]
struct ModelResponse {
    model_id: String,
    content: String,
    error_message: Option<String>,
    metrics: Option<ResponseMetrics>,
}

#[derive(Clone, Debug, PartialEq)]
struct ResponseMetrics {
    request_sent_at: std::time::Instant,
    first_token_at: Option<std::time::Instant>,
    completed_at: Option<std::time::Instant>,
}

impl ResponseMetrics {
    fn time_to_first_token(&self) -> Option<std::time::Duration> {
        self.first_token_at.map(|ft| ft.duration_since(self.request_sent_at))
    }
    
    fn total_time(&self) -> Option<std::time::Duration> {
        self.completed_at.map(|ct| ct.duration_since(self.request_sent_at))
    }
    
    fn format_duration(duration: std::time::Duration) -> String {
        let millis = duration.as_millis();
        if millis < 1000 {
            format!("{}ms", millis)
        } else {
            let secs = duration.as_secs_f64();
            if secs < 60.0 {
                format!("{:.2}s", secs)
            } else {
                let mins = secs / 60.0;
                format!("{:.1}m", mins)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ConversationHistory {
    // For single model: Vec<(user_msg, assistant_msg)>
    single_model: Vec<(String, String)>,
    // For multi-model: HashMap<model_id, Vec<(user_msg, assistant_msg)>>
    multi_model: HashMap<String, Vec<(String, String)>>,
}

#[derive(Props, Clone)]
pub struct StandardProps {
    theme: Signal<Theme>,
    client: Option<Arc<OpenRouterClient>>,
    input_settings: Signal<InputSettings>,
    session_id: Option<String>,
    on_session_saved: EventHandler<ChatSession>,
}

impl PartialEq for StandardProps {
    fn eq(&self, other: &Self) -> bool {
        self.theme == other.theme 
            && self.input_settings == other.input_settings
            && self.session_id == other.session_id
        // Skip client and callback comparison
    }
}

#[component]
pub fn Standard(props: StandardProps) -> Element {
    let theme = props.theme;
    let client = props.client.clone();
    let client_for_send = props.client;
    let input_settings = props.input_settings;
    let active_runs = use_context::<Signal<HashMap<String, ActiveRunRecord>>>();
    let _ = theme.read();
    let mut selected_models = use_signal(|| Vec::<String>::new());
    let mut user_messages = use_signal(|| Vec::<String>::new());
    let mut model_responses = use_signal(|| Vec::<Vec<ModelResponse>>::new());
    #[derive(Clone, Debug)]
    struct StreamingResponse {
        content: String,
        metrics: ResponseMetrics,
    }
    
    let mut current_streaming_responses = use_signal(|| HashMap::<String, StreamingResponse>::new());
    let mut is_streaming = use_signal(|| false);
    let mut current_run_id = use_signal(|| None::<String>);
    
    // System prompt state
    let mut system_prompt = use_signal(|| "You are a helpful AI assistant.".to_string());
    let mut system_prompt_editor_open = use_signal(|| false);
    let mut temp_system_prompt = use_signal(|| String::new());
    
    // Conversation history (per model for multi-model mode)
    let mut conversation_history = use_signal(|| ConversationHistory {
        single_model: Vec::new(),
        multi_model: HashMap::new(),
    });
    
    // Track the currently loaded session to avoid reloading on every render
    let mut loaded_session_id = use_signal(|| None::<String>);
    
    // Load history if session_id changes
    let session_id = props.session_id.clone();
    
    // Use resource for async loading
    let _session_loader = use_resource(move || {
        let session_id = session_id.clone();
        async move {
            if let Some(sid) = session_id {
                // Run file I/O in a blocking task
                let result = tokio::task::spawn_blocking(move || ChatHistory::load_session(&sid)).await;
                match result {
                    Ok(Ok(session_data)) => Some(Ok(session_data)),
                    Ok(Err(e)) => Some(Err(e)),
                    Err(e) => Some(Err(format!("Task join error: {}", e))),
                }
            } else {
                None
            }
        }
    });

    // Update state when resource is ready
    if let Some(Some(result)) = _session_loader.read().as_ref() {
        let current_sid = props.session_id.clone();
        if current_sid != *loaded_session_id.read() {
            match result {
                Ok(session_data) => {
                    if let ChatHistory::Standard(history) = &session_data.history {
                        loaded_session_id.set(current_sid);
                        selected_models.set(history.selected_models.clone());
                        user_messages.set(history.user_messages.clone());
                        system_prompt.set(history.system_prompt.clone());
                        
                        // Convert ModelResponse from history to internal format
                        let converted_responses: Vec<Vec<ModelResponse>> = history.model_responses
                            .iter()
                            .map(|responses| {
                                responses.iter()
                                    .map(|r| ModelResponse {
                                        model_id: r.model_id.clone(),
                                        content: r.content.clone(),
                                        error_message: r.error_message.clone(),
                                        metrics: None, // Historical responses don't have metrics
                                    })
                                    .collect()
                            })
                            .collect();
                        model_responses.set(converted_responses);
                        
                        // Convert ConversationHistory
                        conversation_history.set(ConversationHistory {
                            single_model: history.conversation_history.single_model.clone(),
                            multi_model: history.conversation_history.multi_model.clone(),
                        });
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load session: {}", e);
                    loaded_session_id.set(current_sid);
                    selected_models.set(Vec::new());
                    user_messages.set(Vec::new());
                    model_responses.set(Vec::new());
                    system_prompt.set("You are a helpful AI assistant.".to_string());
                    conversation_history.set(ConversationHistory {
                        single_model: Vec::new(),
                        multi_model: HashMap::new(),
                    });
                }
            }
        }
    } else if props.session_id.is_none() && loaded_session_id.read().is_some() {
        // Reset for new session
        loaded_session_id.set(None);
        selected_models.set(Vec::new());
        user_messages.set(Vec::new());
        model_responses.set(Vec::new());
        system_prompt.set("You are a helpful AI assistant.".to_string());
        conversation_history.set(ConversationHistory {
            single_model: Vec::new(),
            multi_model: HashMap::new(),
        });
    }
    

    // Handle model selection
    let on_models_selected = move |models: Vec<String>| {
        selected_models.set(models.clone());
        // Initialize conversation history for each model
        let mut history = conversation_history.write();
        history.multi_model.clear();
        for model_id in &models {
            history.multi_model.insert(model_id.clone(), Vec::new());
        }
    };
    
    // System prompt editor handlers
    let open_system_prompt_editor = move |_| {
        temp_system_prompt.set(system_prompt());
        system_prompt_editor_open.set(true);
    };
    
    let save_system_prompt = move |_| {
        system_prompt.set(temp_system_prompt());
        system_prompt_editor_open.set(false);
    };

    let active_run_for_session = find_run_for_session(active_runs, &props.session_id, ChatMode::Standard);
    let run_is_active = active_run_for_session.as_ref().is_some_and(|run| {
        matches!(run.status, RunStatus::Running | RunStatus::Cancelling)
    });
    let cancel_bar_run = active_run_for_session.as_ref().and_then(|run| {
        if matches!(run.status, RunStatus::Running | RunStatus::Cancelling) {
            Some((run.id.clone(), matches!(run.status, RunStatus::Cancelling)))
        } else {
            None
        }
    });
    if let Some(active_run) = &active_run_for_session {
        if current_run_id.read().as_ref() != Some(&active_run.id) {
            current_run_id.set(Some(active_run.id.clone()));
        }
    } else if current_run_id.read().is_some() {
        current_run_id.set(None);
    }

    // Handle sending a message
    let send_message = move |text: String| {
        if text.trim().is_empty() || *is_streaming.read() || run_is_active {
            return;
        }

        let models = selected_models.read().clone();
        if models.is_empty() {
            return;
        }

        // Add user message
        user_messages.write().push(text.clone());

        // Start streaming from all selected models
        if let Some(client_arc) = &client_for_send {
            let client = client_arc.clone();
            let is_single_model = models.len() == 1;
            let sys_prompt = system_prompt();
            let mut is_streaming_clone = is_streaming.clone();
            let mut current_streaming_responses_clone = current_streaming_responses.clone();
            let mut model_responses_clone = model_responses.clone();
            let mut conversation_history_clone = conversation_history.clone();
            let session_id_for_save = props.session_id.clone();
            let on_session_saved = props.on_session_saved.clone();
            let user_messages_save = user_messages.clone();
            let selected_models_save = selected_models.clone();
            let system_prompt_save = system_prompt.clone();
            let cancel_flag = Arc::new(AtomicBool::new(false));
            let run_id = create_run_id(ChatMode::Standard, &props.session_id);
            current_run_id.set(Some(run_id.clone()));

            let run_id_for_task = run_id.clone();
            let mut active_runs_for_task = active_runs.clone();
            let cancel_flag_for_task = cancel_flag.clone();
            let task = spawn_forever(async move {
                try_signal_set(&mut is_streaming_clone, true);
                let _ = try_signal_update(&mut current_streaming_responses_clone, |responses| {
                    responses.clear()
                });

                // For single model, use its history directly
                // For multiple models, we need to stream each separately with their own history
                // Since we can't use stream_chat_completion_multi with different messages per model,
                // we'll stream each model individually and aggregate results
                
                let mut final_results: HashMap<String, (String, Option<String>, Option<ResponseMetrics>)> = HashMap::new();
                
                if is_single_model {
                    // Single model with shared history
                    let history = try_signal_read(&conversation_history_clone, |history| history.clone())
                        .unwrap_or(ConversationHistory {
                            single_model: Vec::new(),
                            multi_model: HashMap::new(),
                        });
                    let mut messages = vec![ChatMessage::system(sys_prompt.clone())];
                    for (user_msg, assistant_msg) in &history.single_model {
                        messages.push(ChatMessage::user(user_msg.clone()));
                        messages.push(ChatMessage::assistant(assistant_msg.clone()));
                    }
                    messages.push(ChatMessage::user(text.clone()));
                    
                    let model_id = models[0].clone();
                    let request_sent_at = std::time::Instant::now();
                    let mut first_token_received = false;
                    
                    match client.stream_chat_completion(model_id.clone(), messages).await {
                        Ok(mut stream) => {
                            let mut content = String::new();
                            
                            // Initialize metrics
                            let mut metrics = ResponseMetrics {
                                request_sent_at,
                                first_token_at: None,
                                completed_at: None,
                            };
                            
                            // Throttle updates: only write to signal every 16ms
                            let mut last_update = std::time::Instant::now();
                            const UPDATE_INTERVAL_MS: u64 = 50; // ~20fps
                            
                            while let Some(event) = next_stream_event_with_cancel(&mut stream, &cancel_flag_for_task).await {
                                match event {
                                    StreamEvent::Content(chunk) => {
                                        // Track first token
                                        if !first_token_received {
                                            metrics.first_token_at = Some(std::time::Instant::now());
                                            first_token_received = true;
                                        }
                                        
                                        content.push_str(&chunk);
                                        
                                    // Throttle UI updates to reduce re-render churn.
                                        if last_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS as u128 {
                                            let _ = try_signal_update(&mut current_streaming_responses_clone, |responses| {
                                                responses.insert(model_id.clone(), StreamingResponse {
                                                    content: content.clone(),
                                                    metrics: metrics.clone(),
                                                });
                                            });
                                            last_update = std::time::Instant::now();
                                        }
                                    }
                                    StreamEvent::Done => {
                                        metrics.completed_at = Some(std::time::Instant::now());
                                        final_results.insert(model_id.clone(), (content.clone(), None, Some(metrics)));
                                        break;
                                    }
                                    StreamEvent::Error(e) => {
                                        if e == "Cancelled" {
                                            break;
                                        }
                                        metrics.completed_at = Some(std::time::Instant::now());
                                        let error_msg = format!("Error: {}", e);
                                        // Immediately show error in streaming UI
                                        let _ = try_signal_update(&mut current_streaming_responses_clone, |responses| {
                                            responses.insert(model_id.clone(), StreamingResponse {
                                                content: error_msg.clone(),
                                                metrics: metrics.clone(),
                                            });
                                        });
                                        final_results.insert(model_id.clone(), (String::new(), Some(e), Some(metrics)));
                                        break;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let metrics = ResponseMetrics {
                                request_sent_at,
                                first_token_at: None,
                                completed_at: Some(std::time::Instant::now()),
                            };
                            let error_msg = format!("Error: {}", e);
                            // Immediately show error in streaming UI
                            let _ = try_signal_update(&mut current_streaming_responses_clone, |responses| {
                                responses.insert(model_id.clone(), StreamingResponse {
                                    content: error_msg.clone(),
                                    metrics: metrics.clone(),
                                });
                            });
                            final_results.insert(model_id, (String::new(), Some(e), Some(metrics)));
                        }
                    }
                } else {
                    // Multiple models, each with separate history - stream concurrently
                    use std::sync::Arc;
                    use tokio::sync::Mutex;
                    use futures::future::join_all;
                    
                    let shared_results = Arc::new(Mutex::new(HashMap::new()));
                    let mut futures = Vec::new();
                    
                    for model_id in &models {
                        let client = client.clone();
                        let model_id = model_id.clone();
                        let sys_prompt = sys_prompt.clone();
                        let text = text.clone();
                        let conversation_history_clone = conversation_history_clone.clone();
                        let mut current_streaming_responses_clone = current_streaming_responses_clone.clone();
                        let shared_results = shared_results.clone();
                        let cancel_flag_for_model = cancel_flag_for_task.clone();
                        
                        let future = async move {
                            let request_sent_at = std::time::Instant::now();
                            let mut first_token_received = false;
                            
                            let history = try_signal_read(&conversation_history_clone, |history| history.clone())
                                .unwrap_or(ConversationHistory {
                                    single_model: Vec::new(),
                                    multi_model: HashMap::new(),
                                });
                            let mut messages = vec![ChatMessage::system(sys_prompt)];
                            if let Some(model_history) = history.multi_model.get(&model_id) {
                                for (user_msg, assistant_msg) in model_history {
                                    messages.push(ChatMessage::user(user_msg.clone()));
                                    messages.push(ChatMessage::assistant(assistant_msg.clone()));
                                }
                            }
                            messages.push(ChatMessage::user(text));
                            
                            match client.stream_chat_completion(model_id.clone(), messages).await {
                                Ok(mut stream) => {
                                    let mut content = String::new();
                                    
                                    // Initialize metrics
                                    let mut metrics = ResponseMetrics {
                                        request_sent_at,
                                        first_token_at: None,
                                        completed_at: None,
                                    };
                                    
                                    // Buffer content locally to throttle updates
                                    let mut last_update = std::time::Instant::now();
                                    const UPDATE_INTERVAL_MS: u64 = 50; // ~20fps
                                    
                                    while let Some(event) = next_stream_event_with_cancel(&mut stream, &cancel_flag_for_model).await {
                                        match event {
                                            StreamEvent::Content(chunk) => {
                                                // Track first token
                                                if !first_token_received {
                                                    metrics.first_token_at = Some(std::time::Instant::now());
                                                    first_token_received = true;
                                                }
                                                
                                                content.push_str(&chunk);
                                                
                                    // Throttle UI updates to reduce re-render churn.
                                                if last_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS as u128 {
                                                    let _ = try_signal_update(&mut current_streaming_responses_clone, |responses| {
                                                        responses.insert(model_id.clone(), StreamingResponse {
                                                            content: content.clone(),
                                                            metrics: metrics.clone(),
                                                        });
                                                    });
                                                    last_update = std::time::Instant::now();
                                                }
                                            }
                                            StreamEvent::Done => {
                                                metrics.completed_at = Some(std::time::Instant::now());
                                                // Flush final content
                                                let _ = try_signal_update(&mut current_streaming_responses_clone, |responses| {
                                                    responses.insert(model_id.clone(), StreamingResponse {
                                                        content: content.clone(),
                                                        metrics: metrics.clone(),
                                                    });
                                                });
                                                shared_results.lock().await.insert(model_id.clone(), (content, None, Some(metrics)));
                                                break;
                                            }
                                            StreamEvent::Error(e) => {
                                                if e == "Cancelled" {
                                                    break;
                                                }
                                                metrics.completed_at = Some(std::time::Instant::now());
                                                let error_msg = format!("Error: {}", e);
                                                // Immediately show error in streaming UI
                                                let _ = try_signal_update(&mut current_streaming_responses_clone, |responses| {
                                                    responses.insert(model_id.clone(), StreamingResponse {
                                                        content: error_msg.clone(),
                                                        metrics: metrics.clone(),
                                                    });
                                                });
                                                shared_results.lock().await.insert(model_id.clone(), (String::new(), Some(e), Some(metrics)));
                                                break;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let metrics = ResponseMetrics {
                                        request_sent_at,
                                        first_token_at: None,
                                        completed_at: Some(std::time::Instant::now()),
                                    };
                                    let error_msg = format!("Error: {}", e);
                                    // Immediately show error in streaming UI
                                    let _ = try_signal_update(&mut current_streaming_responses_clone, |responses| {
                                        responses.insert(model_id.clone(), StreamingResponse {
                                            content: error_msg.clone(),
                                            metrics: metrics.clone(),
                                        });
                                    });
                                    shared_results.lock().await.insert(model_id.clone(), (String::new(), Some(e), Some(metrics)));
                                }
                            }
                        };
                        
                        futures.push(future);
                    }
                    
                    // Run all streams concurrently
                    join_all(futures).await;
                    
                    // Extract final results from Arc<Mutex>
                    let locked_results = shared_results.lock().await;
                    for (k, v) in locked_results.iter() {
                        final_results.insert(k.clone(), v.clone());
                    }
                }
                
                // Build final responses
                let mut final_responses: Vec<ModelResponse> = models
                    .iter()
                    .map(|model_id| {
                        let (content, error, metrics) = final_results.get(model_id)
                            .cloned()
                            .unwrap_or_else(|| (String::new(), Some("No response received".to_string()), None));
                        ModelResponse {
                            model_id: model_id.clone(),
                            content,
                            error_message: error,
                            metrics,
                        }
                    })
                    .collect();
                
                // Update conversation history
                let _ = try_signal_update(&mut conversation_history_clone, |history| {
                    if is_single_model {
                        if let Some(response) = final_responses.first() {
                            if response.error_message.is_none() {
                                history.single_model.push((text.clone(), response.content.clone()));
                            }
                        }
                    } else {
                        for response in &final_responses {
                            if response.error_message.is_none() {
                                if let Some(model_history) = history.multi_model.get_mut(&response.model_id) {
                                    model_history.push((text.clone(), response.content.clone()));
                                }
                            }
                        }
                    }
                });
                
                let _ = try_signal_update(&mut model_responses_clone, |responses| {
                    responses.push(final_responses)
                });
                let _ = try_signal_update(&mut current_streaming_responses_clone, |responses| {
                    responses.clear()
                });
                try_signal_set(&mut is_streaming_clone, false);
                
                // Auto-save only when there is content (spawn_blocking + cloned signals for current state)
                if let Some(sid) = session_id_for_save {
                    let history = StandardHistory {
                        user_messages: try_signal_read(&user_messages_save, |messages| messages.clone())
                            .unwrap_or_default(),
                        model_responses: try_signal_read(&model_responses_clone, |responses| responses.clone())
                            .unwrap_or_default()
                            .iter()
                            .map(|responses| {
                                responses.iter()
                                    .map(|r| crate::utils::ModelResponse {
                                        model_id: r.model_id.clone(),
                                        content: r.content.clone(),
                                        error_message: r.error_message.clone(),
                                    })
                                    .collect()
                            })
                            .collect(),
                        selected_models: try_signal_read(&selected_models_save, |models| models.clone())
                            .unwrap_or_default(),
                        system_prompt: try_signal_read(&system_prompt_save, |prompt| prompt.clone())
                            .unwrap_or_default(),
                        conversation_history: crate::utils::ConversationHistory {
                            single_model: try_signal_read(&conversation_history_clone, |history| {
                                history.single_model.clone()
                            })
                            .unwrap_or_default(),
                            multi_model: try_signal_read(&conversation_history_clone, |history| {
                                history.multi_model.clone()
                            })
                            .unwrap_or_default(),
                        },
                    };
                    let history_enum = ChatHistory::Standard(history.clone());
                    if !ChatHistory::has_content(&history_enum) {
                        return;
                    }
                    let summary = ChatHistory::generate_chat_summary(&history_enum);
                    let session = ChatSession {
                        id: sid.clone(),
                        title: summary,
                        mode: ChatMode::Standard,
                        timestamp: ChatHistory::format_timestamp(),
                    };
                    let session_data = SessionData {
                        session: session.clone(),
                        history: history_enum,
                        created_at: ChatHistory::session_timestamp_from_id(&sid)
                            .unwrap_or_else(ChatHistory::format_timestamp),
                        updated_at: ChatHistory::format_timestamp(),
                    };
                    match tokio::task::spawn_blocking(move || ChatHistory::save_session(&session_data)).await {
                        Err(e) => eprintln!("Failed to save session task: {}", e),
                        Ok(Err(e)) => eprintln!("Failed to save session: {}", e),
                        Ok(Ok(_)) => on_session_saved.call(session),
                    }
                }
                if cancel_flag_for_task.load(Ordering::SeqCst) {
                    set_run_status(active_runs_for_task, &run_id_for_task, RunStatus::Cancelled);
                } else {
                    remove_run(active_runs_for_task, &run_id_for_task);
                }
            });

            register_active_run(
                active_runs,
                run_id,
                props.session_id.clone(),
                ChatMode::Standard,
                "Standard response".to_string(),
                task,
                cancel_flag,
            );
        }
    };

    rsx! {
        div {
            class: "flex flex-col h-full",

            // Show model selector if no models selected
            if selected_models.read().is_empty() {
                if let Some(client_arc) = &client {
                    ModelSelector {
                        theme,
                        client: client_arc.clone(),
                        on_models_selected,
                    }
                } else {
                    div {
                        class: "flex items-center justify-center h-full",
                        div {
                            class: "text-center",
                            p {
                                class: "text-[var(--color-base-content)]/70",
                                "No API client available"
                            }
                        }
                    }
                }
            } else {
                // System prompt header
                div {
                    class: "p-3 border-b border-[var(--color-base-300)] bg-[var(--color-base-100)]",
                    div {
                        class: "flex items-center justify-between",
                        h3 {
                            class: "text-sm font-semibold text-[var(--color-base-content)]",
                            "System Prompt"
                        }
                        div {
                            class: "flex items-center gap-2",
                            button {
                                onclick: open_system_prompt_editor,
                                class: "text-xs text-[var(--color-primary)] hover:underline",
                                "Edit"
                            }
                            button {
                                onclick: move |_| selected_models.set(Vec::new()),
                                class: "text-xs text-[var(--color-primary)] hover:underline",
                                "Change Models"
                            }
                        }
                    }
                    div {
                        class: "text-xs text-[var(--color-base-content)]/70 mt-1 truncate",
                        "{system_prompt}"
                    }
                }
                
                // Chat interface
                div {
                    class: "flex-1 min-h-0 overflow-y-auto p-4",

                    if user_messages.read().is_empty() {
                        // Empty state
                        div {
                            class: "flex flex-col items-center justify-center h-full",
                            div {
                                class: "text-4xl mb-4",
                                "💬"
                            }
                            h2 {
                                class: "text-xl font-bold text-[var(--color-base-content)] mb-2",
                                "Ready to Chat"
                            }
                            p {
                                class: "text-sm text-[var(--color-base-content)]/70 mb-4",
                                "{selected_models.read().len()} model(s) selected"
                            }
                            button {
                                onclick: move |_| selected_models.set(Vec::new()),
                                class: "text-sm text-[var(--color-primary)] hover:underline",
                                "Change Models"
                            }
                        }
                    } else {
                        // Messages display
                        div {
                            class: "space-y-6 w-full",

                            for (idx, user_msg) in user_messages.read().iter().enumerate() {
                                div {
                                    key: "{idx}",

                                    // User message
                                    div {
                                        class: "flex justify-end mb-4",
                                        div {
                                            class: "max-w-[85%] bg-[var(--color-primary)] text-[var(--color-primary-content)] px-3 sm:px-4 md:px-5 py-2 sm:py-3 rounded-lg text-sm sm:text-base",
                                            FormattedText {
                                                theme,
                                                content: user_msg.clone(),
                                            }
                                        }
                                    }

                                    // Model responses
                                    if let Some(responses) = model_responses.read().get(idx) {
                                        if selected_models.read().len() == 1 {
                                            // Single model - traditional chat display
                                            div {
                                                class: "flex justify-start",
                                                div {
                                                    class: if responses[0].error_message.is_some() {
                                                        "w-full max-w-[85%] bg-red-500/10 border-2 border-red-500/50 text-[var(--color-base-content)] px-3 sm:px-4 md:px-5 py-2 sm:py-3 rounded-lg text-sm sm:text-base"
                                                    } else {
                                                        "w-full max-w-[85%] bg-[var(--color-base-200)] text-[var(--color-base-content)] px-3 sm:px-4 md:px-5 py-2 sm:py-3 rounded-lg text-sm sm:text-base"
                                                    },
                                                    div {
                                                        class: "text-xs text-[var(--color-base-content)]/60 mb-2 flex items-center gap-1",
                                                        if responses[0].error_message.is_some() {
                                                            span { "⚠️" }
                                                        }
                                                        span { "{responses[0].model_id}" }
                                                    }
                                                    if let Some(error) = &responses[0].error_message {
                                                        div {
                                                            class: "text-sm text-[var(--color-base-content)] p-3 bg-red-500/20 rounded",
                                                            div {
                                                                class: "whitespace-pre-wrap mb-2",
                                                                "{error}"
                                                            }
                                                            if error.contains("data policy") || error.contains("data retention") {
                                                                a {
                                                                    href: "https://openrouter.ai/settings/privacy",
                                                                    target: "_blank",
                                                                    class: "text-xs text-[var(--color-primary)] hover:underline",
                                                                    "Configure Privacy Settings →"
                                                                }
                                                            }
                                                        }
                                                    } else {
                                                        div {
                                                            FormattedText {
                                                                theme,
                                                                content: responses[0].content.clone(),
                                                            }
                                                        }
                                                        if let Some(metrics) = &responses[0].metrics {
                                                            div {
                                                                class: "mt-2 pt-2 border-t border-[var(--color-base-300)] text-xs text-[var(--color-base-content)]/60 flex flex-wrap gap-2",
                                                                if let Some(ttft) = metrics.time_to_first_token() {
                                                                    span {
                                                                        class: "flex items-center gap-1",
                                                                        span { "⚡" }
                                                                        span { "TTFT: {ResponseMetrics::format_duration(ttft)}" }
                                                                    }
                                                                }
                                                                if let Some(total) = metrics.total_time() {
                                                                    span {
                                                                        class: "flex items-center gap-1",
                                                                        span { "⏱️" }
                                                                        span { "Total: {ResponseMetrics::format_duration(total)}" }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            // Multiple models - card grid
                                            div {
                                                class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3 w-full",
                                                for response in responses {
                                                    div {
                                                        key: "{response.model_id}",
                                                        class: if response.error_message.is_some() {
                                                            "bg-red-500/10 rounded-lg p-3 sm:p-4 border-2 border-red-500/50 h-full flex flex-col"
                                                        } else {
                                                            "bg-[var(--color-base-200)] rounded-lg p-3 sm:p-4 border border-[var(--color-base-300)] h-full flex flex-col"
                                                        },
                                                        div {
                                                            class: "text-xs font-semibold text-[var(--color-base-content)]/70 mb-2 truncate flex items-center gap-1",
                                                            if response.error_message.is_some() {
                                                                span { "⚠️" }
                                                            }
                                                            span { "{response.model_id}" }
                                                        }
                                                        if let Some(error) = &response.error_message {
                                                            div {
                                                                class: "text-xs text-[var(--color-base-content)] p-2 bg-red-500/20 rounded",
                                                                div {
                                                                    class: "whitespace-pre-wrap mb-2",
                                                                    "{error}"
                                                                }
                                                                if error.contains("data policy") || error.contains("data retention") {
                                                                    a {
                                                                        href: "https://openrouter.ai/settings/privacy",
                                                                        target: "_blank",
                                                                        class: "text-xs text-[var(--color-primary)] hover:underline",
                                                                        "Configure Privacy Settings →"
                                                                    }
                                                                }
                                                            }
                                                        } else {
                                                            div {
                                                                class: "text-sm sm:text-base text-[var(--color-base-content)] flex-1",
                                                                FormattedText {
                                                                    theme,
                                                                    content: response.content.clone(),
                                                                }
                                                            }
                                                            if let Some(metrics) = &response.metrics {
                                                                div {
                                                                    class: "mt-2 pt-2 border-t border-[var(--color-base-300)] text-xs text-[var(--color-base-content)]/60 flex flex-wrap gap-2",
                                                                    if let Some(ttft) = metrics.time_to_first_token() {
                                                                        span {
                                                                            class: "flex items-center gap-1",
                                                                            span { "⚡" }
                                                                            span { "TTFT: {ResponseMetrics::format_duration(ttft)}" }
                                                                        }
                                                                    }
                                                                    if let Some(total) = metrics.total_time() {
                                                                        span {
                                                                            class: "flex items-center gap-1",
                                                                            span { "⏱️" }
                                                                            span { "Total: {ResponseMetrics::format_duration(total)}" }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else if idx == user_messages.read().len() - 1 && *is_streaming.read() {
                                        // Currently streaming responses
                                        {
                                            let streaming_responses = current_streaming_responses.read();
                                            let is_single_model = selected_models.read().len() == 1;
                                            let models = selected_models.read().clone();
                                            
                                            // Check if single model has error
                                            let single_model_is_error = if is_single_model {
                                                streaming_responses.get(&models[0])
                                                    .map(|sr| sr.content.starts_with("Error: "))
                                                    .unwrap_or(false)
                                            } else {
                                                false
                                            };

                                            // Pre-compute error message and privacy link for single model
                                            let (single_error_msg, single_has_privacy) = if is_single_model {
                                                if let Some(streaming) = streaming_responses.get(&models[0]) {
                                                    if streaming.content.starts_with("Error: ") {
                                                        let error_msg = streaming.content.strip_prefix("Error: ").unwrap_or(&streaming.content);
                                                        let has_privacy = error_msg.contains("data policy") || error_msg.contains("data retention");
                                                        (Some(error_msg.to_string()), has_privacy)
                                                    } else {
                                                        (None, false)
                                                    }
                                                } else {
                                                    (None, false)
                                                }
                                            } else {
                                                (None, false)
                                            };

                                            // Pre-compute error messages and privacy flags for all models
                                            let mut model_errors: HashMap<String, (String, bool)> = HashMap::new();
                                            for model_id in models.iter() {
                                                if let Some(streaming) = streaming_responses.get(model_id) {
                                                    if streaming.content.starts_with("Error: ") {
                                                        let error_msg = streaming.content.strip_prefix("Error: ").unwrap_or(&streaming.content).to_string();
                                                        let has_privacy = error_msg.contains("data policy") || error_msg.contains("data retention");
                                                        model_errors.insert(model_id.clone(), (error_msg, has_privacy));
                                                    }
                                                }
                                            }

                                            rsx! {
                                                if is_single_model {
                                                    // Single model streaming
                                                    div {
                                                        class: "flex justify-start",
                                                        div {
                                                            class: if single_model_is_error {
                                                                "w-full max-w-[85%] bg-red-500/10 border-2 border-red-500/50 text-[var(--color-base-content)] px-3 sm:px-4 md:px-5 py-2 sm:py-3 rounded-lg text-sm sm:text-base"
                                                            } else {
                                                                "w-full max-w-[85%] bg-[var(--color-base-200)] text-[var(--color-base-content)] px-4 py-3 rounded-lg"
                                                            },
                                                            div {
                                                                class: "text-xs text-[var(--color-base-content)]/60 mb-2 flex items-center gap-2",
                                                                if single_model_is_error {
                                                                    span { "⚠️" }
                                                                }
                                                                span { "{models[0]}" }
                                                                if !single_model_is_error {
                                                                    span {
                                                                        class: "inline-block w-2 h-2 bg-[var(--color-primary)] rounded-full animate-pulse"
                                                                    }
                                                                }
                                                            }
                                                            if let Some(streaming) = streaming_responses.get(&models[0]) {
                                                                if streaming.content.starts_with("Error: ") {
                                                                    if let Some(ref error_msg) = single_error_msg {
                                                                        div {
                                                                            class: "text-sm text-[var(--color-base-content)] p-3 bg-red-500/20 rounded",
                                                                            div {
                                                                                class: "whitespace-pre-wrap mb-2",
                                                                                "{error_msg}"
                                                                            }
                                                                            if single_has_privacy {
                                                                                a {
                                                                                    href: "https://openrouter.ai/settings/privacy",
                                                                                    target: "_blank",
                                                                                    class: "text-xs text-[var(--color-primary)] hover:underline",
                                                                                    "Configure Privacy Settings →"
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                } else {
                                                            div {
                                                                class: "whitespace-pre-wrap break-words",
                                                                "{streaming.content}"
                                                            }
                                                        }
                                                    }
                                                    if let Some(streaming) = streaming_responses.get(&models[0]) {
                                                        // Only show metrics if not an error (errors complete immediately)
                                                        if !streaming.content.starts_with("Error: ") {
                                                            div {
                                                                class: "mt-2 pt-2 border-t border-[var(--color-base-300)] text-xs text-[var(--color-base-content)]/60 flex flex-wrap gap-2",
                                                                if let Some(ttft) = streaming.metrics.time_to_first_token() {
                                                                    span {
                                                                        class: "flex items-center gap-1",
                                                                        span { "⚡" }
                                                                        span { "TTFT: {ResponseMetrics::format_duration(ttft)}" }
                                                                    }
                                                                }
                                                                span {
                                                                    class: "flex items-center gap-1",
                                                                    span { "🔄" }
                                                                    span { "Streaming..." }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    }
                                                }
                                            } else {
                                                // Multiple models streaming
                                                div {
                                                class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3 w-full",
                                                for model_id in models.iter() {
                                                    div {
                                                        key: "{model_id}",
                                                        class: if streaming_responses.get(model_id).map(|sr| sr.content.starts_with("Error: ")).unwrap_or(false) {
                                                            "bg-red-500/10 rounded-lg p-3 sm:p-4 border-2 border-red-500/50 h-full flex flex-col"
                                                        } else {
                                                            "bg-[var(--color-base-200)] rounded-lg p-3 sm:p-4 border border-[var(--color-base-300)] h-full flex flex-col"
                                                        },
                                                        div {
                                                            class: "text-xs font-semibold text-[var(--color-base-content)]/70 mb-2 truncate flex items-center gap-2",
                                                            if streaming_responses.get(model_id).map(|sr| sr.content.starts_with("Error: ")).unwrap_or(false) {
                                                                span { "⚠️" }
                                                            }
                                                            span { "{model_id}" }
                                                            if !streaming_responses.get(model_id).map(|sr| sr.content.starts_with("Error: ")).unwrap_or(false) {
                                                                span {
                                                                    class: "inline-block w-2 h-2 bg-[var(--color-primary)] rounded-full animate-pulse"
                                                                }
                                                            }
                                                        }
                                                        if let Some(streaming) = streaming_responses.get(model_id) {
                                                            if streaming.content.starts_with("Error: ") {
                                                                if let Some((ref error_msg, has_privacy_link)) = model_errors.get(model_id) {
                                                                    div {
                                                                        class: "text-xs text-[var(--color-base-content)] p-2 bg-red-500/20 rounded min-h-[3rem] flex-1",
                                                                        div {
                                                                            class: "whitespace-pre-wrap mb-2",
                                                                            "{error_msg}"
                                                                        }
                                                                        if *has_privacy_link {
                                                                            a {
                                                                                href: "https://openrouter.ai/settings/privacy",
                                                                                target: "_blank",
                                                                                class: "text-xs text-[var(--color-primary)] hover:underline",
                                                                                "Configure Privacy Settings →"
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            } else {
                                                                div {
                                                                    class: "text-sm sm:text-base text-[var(--color-base-content)] min-h-[3rem] flex-1",
                                                                    div {
                                                                        class: "whitespace-pre-wrap break-words",
                                                                        "{streaming.content}"
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        if let Some(streaming) = streaming_responses.get(model_id) {
                                                            // Only show metrics if not an error (errors complete immediately)
                                                            if !streaming.content.starts_with("Error: ") {
                                                                div {
                                                                    class: "mt-2 pt-2 border-t border-[var(--color-base-300)] text-xs text-[var(--color-base-content)]/60 flex flex-wrap gap-2",
                                                                    if let Some(ttft) = streaming.metrics.time_to_first_token() {
                                                                        span {
                                                                            class: "flex items-center gap-1",
                                                                            span { "⚡" }
                                                                            span { "TTFT: {ResponseMetrics::format_duration(ttft)}" }
                                                                        }
                                                                    }
                                                                    span {
                                                                        class: "flex items-center gap-1",
                                                                        span { "🔄" }
                                                                        span { "Streaming..." }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    }
                                                }
                                            }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some((active_run_id, is_cancelling)) = cancel_bar_run.clone() {
                        div {
                            class: "px-4 py-3 border-t border-[var(--color-base-300)] bg-[var(--color-base-100)] flex items-center justify-between gap-3",
                            div {
                                class: "text-sm text-[var(--color-base-content)]/70",
                                if is_cancelling {
                                    "Cancelling active response..."
                                } else {
                                    "A response is running in the background. You can leave this chat open or cancel it manually."
                                }
                            }
                            button {
                                onclick: move |_| {
                                    if let Some(run) = active_runs.read().get(&active_run_id).cloned() {
                                        run.request_cancel();
                                        set_run_status(active_runs, &active_run_id, RunStatus::Cancelling);
                                    }
                                },
                                disabled: is_cancelling,
                                class: "px-3 py-2 rounded-lg bg-red-500 text-white text-sm font-medium hover:bg-red-500/90 disabled:opacity-60 disabled:cursor-not-allowed",
                                if is_cancelling {
                                    "Cancelling..."
                                } else {
                                    "Cancel Run"
                                }
                            }
                        }
                }

                // Input area
                ChatInput {
                    theme,
                    input_settings,
                    on_send: send_message,
                }
            }
            
            // System Prompt Editor Modal
            Modal {
                theme,
                open: system_prompt_editor_open,
                on_close: move |_| {
                    temp_system_prompt.set(system_prompt());
                    system_prompt_editor_open.set(false);
                },
                
                div {
                    class: "p-6",
                    
                    // Header
                    div {
                        class: "flex items-start justify-between mb-4",
                        div {
                            h2 {
                                class: "text-xl font-bold text-[var(--color-base-content)]",
                                "Edit System Prompt"
                            }
                            p {
                                class: "text-sm text-[var(--color-base-content)]/70 mt-1",
                                "The system prompt sets the behavior and personality of the AI assistant."
                            }
                        }
                        button {
                            class: "text-2xl text-[var(--color-base-content)]/70 hover:text-[var(--color-base-content)] transition-colors",
                            onclick: move |_| {
                                temp_system_prompt.set(system_prompt());
                                system_prompt_editor_open.set(false);
                            },
                            "×"
                        }
                    }
                    
                    // Prompt editor textarea
                    div {
                        class: "mb-4",
                        textarea {
                            value: "{temp_system_prompt}",
                            oninput: move |evt| temp_system_prompt.set(evt.value()),
                            rows: "10",
                            class: "w-full p-3 border-2 rounded-lg font-mono text-sm bg-[var(--color-base-100)] text-[var(--color-base-content)] border-[var(--color-base-300)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent resize-y min-h-[200px]",
                            placeholder: "Enter system prompt...",
                            autofocus: true,
                        }
                    }
                    
                    // Character count
                    div {
                        class: "text-xs text-[var(--color-base-content)]/50 mb-4 text-right",
                        "{temp_system_prompt.read().len()} characters"
                    }
                    
                    // Action buttons
                    div {
                        class: "flex justify-between items-center gap-3",
                        button {
                            onclick: move |_| {
                                temp_system_prompt.set("You are a helpful AI assistant.".to_string());
                            },
                            class: "px-4 py-2 text-sm rounded border border-[var(--color-base-300)] bg-[var(--color-base-200)] text-[var(--color-base-content)] hover:bg-[var(--color-base-300)] transition-colors",
                            "Reset to Default"
                        }
                        div {
                            class: "flex gap-2",
                            button {
                                onclick: move |_| {
                                    temp_system_prompt.set(system_prompt());
                                    system_prompt_editor_open.set(false);
                                },
                                class: "px-4 py-2 text-sm rounded border border-[var(--color-base-300)] bg-[var(--color-base-200)] text-[var(--color-base-content)] hover:bg-[var(--color-base-300)] transition-colors",
                                "Cancel"
                            }
                            button {
                                onclick: save_system_prompt,
                                class: "px-4 py-2 text-sm rounded bg-[var(--color-primary)] text-[var(--color-primary-content)] hover:bg-[var(--color-primary)]/90 transition-colors font-medium",
                                "Save Prompt"
                            }
                        }
                    }
                }
            }
        }
    }
}
