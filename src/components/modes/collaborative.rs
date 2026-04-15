use super::common::{ChatInput, FormattedText, PromptCard, PromptEditorModal, PromptType};
use crate::utils::{
    create_run_id, find_run_for_session, next_stream_event_with_cancel,
    recv_multi_event_with_cancel, register_active_run, remove_run, set_run_status,
    try_signal_read, try_signal_set, try_signal_update, ActiveRunRecord, ChatHistory,
    ChatMessage, ChatMode, ChatSession, InputSettings, Model, OpenRouterClient, RunStatus,
    SessionData, StreamEvent, Theme,
};
use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
struct PromptTemplates {
    initial_response: String,
    cross_review: String,
    consensus: String,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        Self {
            initial_response: "You are part of a collaborative AI team working together to answer questions. Provide your best answer to this question:\n\n{user_question}".to_string(),

            cross_review: "Review the following responses from other AI models. Provide constructive feedback on their strengths and areas for improvement.\n\nUser Question: {user_question}\n\nOther responses:\n{other_responses}\n\nProvide your analysis:".to_string(),

            consensus: "Based on all the initial responses and reviews below, synthesize a final collaborative answer that combines the best insights from all models.\n\nUser Question: {user_question}\n\nInitial Responses:\n{initial_responses}\n\nReviews:\n{reviews}\n\nSynthesize the best collaborative answer:".to_string(),
        }
    }
}

impl PromptTemplates {
    fn get(&self, prompt_type: PromptType) -> String {
        match prompt_type {
            PromptType::Initial => self.initial_response.clone(),
            PromptType::Review => self.cross_review.clone(),
            PromptType::Consensus => self.consensus.clone(),
        }
    }

    fn set(&mut self, prompt_type: PromptType, value: String) {
        match prompt_type {
            PromptType::Initial => self.initial_response = value,
            PromptType::Review => self.cross_review = value,
            PromptType::Consensus => self.consensus = value,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ModelResponse {
    model_id: String,
    content: String,
    error_message: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct CollaborativeRound {
    user_question: String,
    phase1_responses: Vec<ModelResponse>,
    phase2_reviews: Vec<ModelResponse>,
    phase3_consensus: Option<ModelResponse>,
    current_phase: CollaborativePhase,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CollaborativePhase {
    Initial,
    Review,
    Consensus,
    Complete,
}

impl CollaborativePhase {
    fn name(&self) -> &'static str {
        match self {
            CollaborativePhase::Initial => "Phase 1: Initial Responses",
            CollaborativePhase::Review => "Phase 2: Cross-Review",
            CollaborativePhase::Consensus => "Phase 3: Consensus",
            CollaborativePhase::Complete => "Complete",
        }
    }

    fn badge_color(&self) -> &'static str {
        match self {
            CollaborativePhase::Initial => "bg-blue-500",
            CollaborativePhase::Review => "bg-purple-500",
            CollaborativePhase::Consensus => "bg-green-500",
            CollaborativePhase::Complete => "bg-gray-500",
        }
    }
}

// ============================================================================
// Component Props
// ============================================================================

#[derive(Props, Clone)]
pub struct CollaborativeProps {
    theme: Signal<Theme>,
    client: Option<Arc<OpenRouterClient>>,
    input_settings: Signal<InputSettings>,
    session_id: Option<String>,
    on_session_saved: EventHandler<ChatSession>,
}

impl PartialEq for CollaborativeProps {
    fn eq(&self, other: &Self) -> bool {
        self.theme == other.theme 
            && self.input_settings == other.input_settings
            && self.session_id == other.session_id
    }
}

// ============================================================================
// Main Component
// ============================================================================

#[component]
pub fn Collaborative(props: CollaborativeProps) -> Element {
    let theme = props.theme;
    let client = props.client.clone();
    let client_for_send = props.client;
    let input_settings = props.input_settings;
    let active_runs = use_context::<Signal<HashMap<String, ActiveRunRecord>>>();
    let _ = theme.read();

    // Prompt templates
    let mut prompt_templates = use_signal(|| PromptTemplates::default());
    let mut prompt_editor_open = use_signal(|| false);
    let mut editing_prompt_type = use_signal(|| PromptType::Initial);
    let mut temp_prompt = use_signal(|| String::new());

    // Model selection state
    let mut selected_models = use_signal(|| Vec::<String>::new());
    let mut selection_step = use_signal(|| 0); // 0 = select models, 1 = chat

    // Model list state
    let available_models = use_signal(|| None::<Result<Vec<Model>, String>>);
    let mut search_query = use_signal(|| String::new());

    // Chat state
    let mut conversation_history = use_signal(|| Vec::<CollaborativeRound>::new());
    let current_phase = use_signal(|| CollaborativePhase::Initial);
    let mut is_processing = use_signal(|| false);
    let current_streaming_responses = use_signal(|| HashMap::<String, String>::new());
    let mut current_run_id = use_signal(|| None::<String>);

    let mut loaded_session_id = use_signal(|| None::<String>);
    let session_id = props.session_id.clone();
    let session_loader = use_resource(move || {
        let session_id = session_id.clone();
        async move {
            if let Some(sid) = session_id {
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

    if let Some(Some(result)) = session_loader.read().as_ref() {
        let current_sid = props.session_id.clone();
        if current_sid != *loaded_session_id.read() {
            match result {
                Ok(session_data) => {
                    if let ChatHistory::Collaborative(history) = &session_data.history {
                        loaded_session_id.set(current_sid.clone());
                        selected_models.set(history.selected_models.clone());
                        let converted_rounds: Vec<CollaborativeRound> = history
                            .rounds
                            .iter()
                            .map(|r| {
                                let phase1_responses: Vec<ModelResponse> = r
                                    .model_responses
                                    .iter()
                                    .map(|mr| ModelResponse {
                                        model_id: mr.model_id.clone(),
                                        content: mr.content.clone(),
                                        error_message: mr.error_message.clone(),
                                    })
                                    .collect();
                                CollaborativeRound {
                                    user_question: r.user_message.clone(),
                                    phase1_responses,
                                    phase2_reviews: vec![],
                                    phase3_consensus: r.final_consensus.as_ref().map(|consensus| ModelResponse {
                                        model_id: "consensus".to_string(),
                                        content: consensus.clone(),
                                        error_message: None,
                                    }),
                                    current_phase: CollaborativePhase::Complete,
                                }
                            })
                            .collect();
                        conversation_history.set(converted_rounds);
                        if !history.selected_models.is_empty() {
                            selection_step.set(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load session: {}", e);
                    loaded_session_id.set(current_sid);
                    selected_models.set(Vec::new());
                    conversation_history.set(Vec::new());
                    prompt_templates.set(PromptTemplates::default());
                    selection_step.set(0);
                }
            }
        }
    } else if props.session_id.is_none() && loaded_session_id.read().is_some() {
        loaded_session_id.set(None);
        selected_models.set(Vec::new());
        conversation_history.set(Vec::new());
        prompt_templates.set(PromptTemplates::default());
        selection_step.set(0);
    }

    // Fetch models on component mount
    let _fetch = use_hook(|| {
        if let Some(client_arc) = &client {
            let client_clone = client_arc.clone();
            let mut models_clone = available_models.clone();
            spawn(async move {
                let result = client_clone.fetch_models().await;
                models_clone.set(Some(result));
            });
        }
    });

    // Toggle model selection
    let mut toggle_model = move |model_id: String| {
        let mut selected = selected_models.write();
        if let Some(pos) = selected.iter().position(|id| id == &model_id) {
            selected.remove(pos);
        } else {
            selected.push(model_id);
        }
    };

    // Start chat
    let start_chat = move |_| {
        if selected_models.read().len() >= 2 {
            selection_step.set(1);
        }
    };

    // Edit prompt handlers
    let mut open_prompt_editor = move |ptype: PromptType| {
        editing_prompt_type.set(ptype);
        temp_prompt.set(prompt_templates.read().get(ptype));
        prompt_editor_open.set(true);
    };

    let save_prompt = move |new_prompt: String| {
        let mut templates = prompt_templates.write();
        templates.set(*editing_prompt_type.read(), new_prompt);
    };

    let active_run_for_session = find_run_for_session(active_runs, &props.session_id, ChatMode::Collaborative);
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

    // Cancel active run when component unmounts
    {
        let mut active_runs = active_runs.clone();
        let current_run_id = current_run_id.clone();
        use_drop(move || {
            if let Some(run_id) = current_run_id.try_read().ok().and_then(|id| id.clone()) {
                if let Some(run) = active_runs.try_read().ok().and_then(|runs| runs.get(&run_id).cloned()) {
                    run.request_cancel();
                    run.task.cancel();
                }
                remove_run(active_runs, &run_id);
            }
        });
    }

    // Send message handler
    let send_message = move |text: String| {
        if text.trim().is_empty() || *is_processing.read() || run_is_active {
            return;
        }

        let models = selected_models.read().clone();
        if models.len() < 2 {
            return;
        }

        if let Some(client_arc) = &client_for_send {
            let client = client_arc.clone();
            let user_msg = text.clone();
            let mut is_processing_clone = is_processing.clone();
            let mut current_phase_clone = current_phase.clone();
            let mut current_streaming_clone = current_streaming_responses.clone();
            let mut conversation_history_clone = conversation_history.clone();
            let templates = prompt_templates.read().clone();
            let session_id_for_save = props.session_id.clone();
            let on_session_saved = props.on_session_saved.clone();
            let selected_models_for_save = selected_models.read().clone();
            let run_id = create_run_id(ChatMode::Collaborative, &props.session_id);
            current_run_id.set(Some(run_id.clone()));
            let cancel_flag = Arc::new(AtomicBool::new(false));

            // Initialize new round
            conversation_history_clone.write().push(CollaborativeRound {
                user_question: user_msg.clone(),
                phase1_responses: vec![],
                phase2_reviews: vec![],
                phase3_consensus: None,
                current_phase: CollaborativePhase::Initial,
            });

            let run_id_for_task = run_id.clone();
            let mut active_runs_for_task = active_runs.clone();
            let cancel_flag_for_task = cancel_flag.clone();
            let task = spawn_forever(async move {
                try_signal_set(&mut is_processing_clone, true);
                try_signal_set(&mut current_phase_clone, CollaborativePhase::Initial);
                let _ = try_signal_update(&mut current_streaming_clone, |responses| responses.clear());

                // ========================================================
                // PHASE 1: Initial Responses (Parallel)
                // ========================================================

                let initial_prompt = templates.initial_response
                    .replace("{user_question}", &user_msg);

                let messages = vec![
                    ChatMessage::system("You are part of a collaborative AI workflow. Follow each phase instruction precisely.".to_string()),
                    ChatMessage::user(initial_prompt),
                ];

                match client.stream_chat_completion_multi(models.clone(), messages).await {
                    Ok(mut rx) => {
                        let mut done_models = std::collections::HashSet::new();
                        let mut phase1_results: HashMap<String, ModelResponse> = HashMap::new();

                        // Buffer content locally to throttle updates
                        let mut content_buffer: HashMap<String, String> = HashMap::new();
                        let mut last_update = std::time::Instant::now();
                        const UPDATE_INTERVAL_MS: u64 = 50; // ~20fps

                        while let Some(event) = recv_multi_event_with_cancel(&mut rx, &cancel_flag_for_task).await {
                            if cancel_flag_for_task.load(Ordering::SeqCst) {
                                break;
                            }
                            let model_id = event.model_id.clone();

                            match event.event {
                                StreamEvent::Content(content) => {
                                    // Accumulate in buffer instead of writing immediately
                                    content_buffer
                                        .entry(model_id.clone())
                                        .and_modify(|s| s.push_str(&content))
                                        .or_insert(content);

                                        // Throttle updates: only write to signal every 16ms
                                        if last_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS as u128 {
                                            if let Some(accumulated) = content_buffer.get(&model_id) {
                                                let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                                    responses.insert(model_id.clone(), accumulated.clone());
                                                });
                                            }
                                            last_update = std::time::Instant::now();
                                        }
                                }
                                StreamEvent::Done => {
                                    // Flush any remaining buffered content before marking done
                                    if let Some(accumulated) = content_buffer.remove(&model_id) {
                                        let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                            responses.insert(model_id.clone(), accumulated.clone());
                                        });
                                    }

                                    let final_content = try_signal_read(&current_streaming_clone, |responses| {
                                        responses.get(&model_id).cloned().unwrap_or_default()
                                    })
                                    .unwrap_or_default();

                                    phase1_results.insert(
                                        model_id.clone(),
                                        ModelResponse {
                                            model_id: model_id.clone(),
                                            content: final_content,
                                            error_message: None,
                                        },
                                    );
                                    done_models.insert(model_id.clone());

                                    if done_models.len() >= models.len() {
                                        break;
                                    }
                                }
                                StreamEvent::Error(e) => {
                                    if e == "Cancelled" {
                                        break;
                                    }
                                    phase1_results.insert(
                                        model_id.clone(),
                                        ModelResponse {
                                            model_id: model_id.clone(),
                                            content: String::new(),
                                            error_message: Some(e),
                                        },
                                    );
                                    done_models.insert(model_id);
                                }
                            }
                        }

                        // Early exit if cancelled after Phase 1
                        if cancel_flag_for_task.load(Ordering::SeqCst) {
                            try_signal_set(&mut is_processing_clone, false);
                            set_run_status(active_runs_for_task, &run_id_for_task, RunStatus::Cancelled);
                            return;
                        }

                        // Update conversation with Phase 1 results
                        let _ = try_signal_update(&mut conversation_history_clone, |history| {
                            if let Some(last_round) = history.last_mut() {
                                last_round.phase1_responses = models
                                    .iter()
                                    .filter_map(|id| phase1_results.get(id).cloned())
                                    .collect();
                            }
                        });

                        let _ = try_signal_update(&mut current_streaming_clone, |responses| responses.clear());

                        // ========================================================
                        // PHASE 2: Cross-Review (Sequential per model)
                        // ========================================================

                        try_signal_set(&mut current_phase_clone, CollaborativePhase::Review);
                        let _ = try_signal_update(&mut conversation_history_clone, |history| {
                            if let Some(last_round) = history.last_mut() {
                                last_round.current_phase = CollaborativePhase::Review;
                            }
                        });

                        let successful_phase1: Vec<_> = phase1_results
                            .values()
                            .filter(|r| r.error_message.is_none())
                            .collect();

                        if successful_phase1.len() >= 2 {
                            let mut phase2_results = Vec::new();

                            for model_id in &models {
                                // Build "other responses" text
                                let other_responses: String = successful_phase1
                                    .iter()
                                    .filter(|r| &r.model_id != model_id)
                                    .map(|r| format!("{}: {}", r.model_id, r.content))
                                    .collect::<Vec<_>>()
                                    .join("\n\n");

                                let review_prompt = templates.cross_review
                                    .replace("{user_question}", &user_msg)
                                    .replace("{other_responses}", &other_responses);

                                let review_messages = vec![
                                    ChatMessage::system("You are part of a collaborative AI workflow. Follow each phase instruction precisely.".to_string()),
                                    ChatMessage::user(review_prompt),
                                ];

                                match client.stream_chat_completion(model_id.clone(), review_messages).await {
                                    Ok(mut stream) => {
                                        let mut review_content = String::new();

                                        // Throttle updates: only write to signal every 16ms
                                        let mut last_update = std::time::Instant::now();
                                        const UPDATE_INTERVAL_MS: u64 = 50; // ~20fps

                                        while let Some(event) = next_stream_event_with_cancel(&mut stream, &cancel_flag_for_task).await {
                                            if cancel_flag_for_task.load(Ordering::SeqCst) {
                                                break;
                                            }
                                            match event {
                                                StreamEvent::Content(content) => {
                                                    review_content.push_str(&content);

                                                    // Throttle updates: only write to signal every 16ms
                                                    if last_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS as u128 {
                                                        let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                                            responses.insert(model_id.clone(), review_content.clone());
                                                        });
                                                        last_update = std::time::Instant::now();
                                                    }
                                                }
                                                StreamEvent::Done => {
                                                    phase2_results.push(ModelResponse {
                                                        model_id: model_id.clone(),
                                                        content: review_content.clone(),
                                                        error_message: None,
                                                    });
                                                    break;
                                                }
                                                StreamEvent::Error(e) => {
                                                    if e == "Cancelled" {
                                                        break;
                                                    }
                                                    phase2_results.push(ModelResponse {
                                                        model_id: model_id.clone(),
                                                        content: String::new(),
                                                        error_message: Some(e),
                                                    });
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        phase2_results.push(ModelResponse {
                                            model_id: model_id.clone(),
                                            content: String::new(),
                                            error_message: Some(e),
                                        });
                                    }
                                }
                                let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                    responses.remove(model_id);
                                });

                                // Early exit if cancelled during Phase 2
                                if cancel_flag_for_task.load(Ordering::SeqCst) {
                                    break;
                                }
                            }

                            // Early exit if cancelled after Phase 2
                            if cancel_flag_for_task.load(Ordering::SeqCst) {
                                let _ = try_signal_update(&mut conversation_history_clone, |history| {
                                    if let Some(last_round) = history.last_mut() {
                                        last_round.phase2_reviews = phase2_results;
                                    }
                                });
                                try_signal_set(&mut is_processing_clone, false);
                                set_run_status(active_runs_for_task, &run_id_for_task, RunStatus::Cancelled);
                                return;
                            }

                            // Update conversation with Phase 2 results
                            let _ = try_signal_update(&mut conversation_history_clone, |history| {
                                if let Some(last_round) = history.last_mut() {
                                    last_round.phase2_reviews = phase2_results;
                                }
                            });

                            // ========================================================
                            // PHASE 3: Consensus Synthesis
                            // ========================================================

                            try_signal_set(&mut current_phase_clone, CollaborativePhase::Consensus);
                            let _ = try_signal_update(&mut conversation_history_clone, |history| {
                                if let Some(last_round) = history.last_mut() {
                                    last_round.current_phase = CollaborativePhase::Consensus;
                                }
                            });

                            // Use first model as synthesizer
                            let synthesizer_id = &models[0];

                            let initial_responses_text: String = successful_phase1
                                .iter()
                                .map(|r| format!("{}: {}", r.model_id, r.content))
                                .collect::<Vec<_>>()
                                .join("\n\n");

                            let reviews_text: String = try_signal_read(&conversation_history_clone, |history| {
                                history.last().map(|round| {
                                    round.phase2_reviews
                                        .iter()
                                        .filter(|r| r.error_message.is_none())
                                        .map(|r| format!("{}: {}", r.model_id, r.content))
                                        .collect::<Vec<_>>()
                                        .join("\n\n")
                                })
                            })
                            .flatten()
                            .unwrap_or_default();

                            let consensus_prompt = templates.consensus
                                .replace("{user_question}", &user_msg)
                                .replace("{initial_responses}", &initial_responses_text)
                                .replace("{reviews}", &reviews_text);

                            let consensus_messages = vec![
                                ChatMessage::system("You are part of a collaborative AI workflow. Follow each phase instruction precisely.".to_string()),
                                ChatMessage::user(consensus_prompt),
                            ];

                            match client.stream_chat_completion(synthesizer_id.clone(), consensus_messages).await {
                                Ok(mut stream) => {
                                    let mut consensus_content = String::new();
                                    
                                    // Throttle updates: only write to signal every 16ms
                                    let mut last_update = std::time::Instant::now();
                                    const UPDATE_INTERVAL_MS: u64 = 50; // ~20fps

                                    while let Some(event) = next_stream_event_with_cancel(&mut stream, &cancel_flag_for_task).await {
                                        if cancel_flag_for_task.load(Ordering::SeqCst) {
                                            break;
                                        }
                                        match event {
                                            StreamEvent::Content(content) => {
                                                consensus_content.push_str(&content);

                                                // Throttle updates: only write to signal every 16ms
                                                if last_update.elapsed().as_millis() >= UPDATE_INTERVAL_MS as u128 {
                                                    let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                                        responses.insert(
                                                            "consensus".to_string(),
                                                            consensus_content.clone(),
                                                        );
                                                    });
                                                    last_update = std::time::Instant::now();
                                                }
                                            }
                                            StreamEvent::Done => {
                                                // Flush final content
                                                let _ = try_signal_update(&mut current_streaming_clone, |responses| {
                                                    responses.insert(
                                                        "consensus".to_string(),
                                                        consensus_content.clone(),
                                                    );
                                                });

                                                let _ = try_signal_update(&mut conversation_history_clone, |history| {
                                                    if let Some(last_round) = history.last_mut() {
                                                        last_round.phase3_consensus = Some(ModelResponse {
                                                            model_id: synthesizer_id.clone(),
                                                            content: consensus_content,
                                                            error_message: None,
                                                        });
                                                        last_round.current_phase = CollaborativePhase::Complete;
                                                    }
                                                });
                                                break;
                                            }
                                            StreamEvent::Error(e) => {
                                                if e == "Cancelled" {
                                                    break;
                                                }
                                                let _ = try_signal_update(&mut conversation_history_clone, |history| {
                                                    if let Some(last_round) = history.last_mut() {
                                                        last_round.phase3_consensus = Some(ModelResponse {
                                                            model_id: synthesizer_id.clone(),
                                                            content: String::new(),
                                                            error_message: Some(e),
                                                        });
                                                    }
                                                });
                                                break;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = try_signal_update(&mut conversation_history_clone, |history| {
                                        if let Some(last_round) = history.last_mut() {
                                            last_round.phase3_consensus = Some(ModelResponse {
                                                model_id: synthesizer_id.clone(),
                                                content: String::new(),
                                                error_message: Some(e),
                                            });
                                        }
                                    });
                                }
                            }
                        }

                        let _ = try_signal_update(&mut current_streaming_clone, |responses| responses.clear());
                        try_signal_set(&mut current_phase_clone, CollaborativePhase::Complete);
                        try_signal_set(&mut is_processing_clone, false);
                        
                        // Auto-save only when there is content (spawn_blocking to avoid blocking async runtime)
                        if let Some(sid) = session_id_for_save {
                            let history_rounds: Vec<crate::utils::CollaborativeRound> = try_signal_read(&conversation_history_clone, |history| history.clone())
                                .unwrap_or_default()
                                .iter()
                                .map(|r| {
                                    let model_responses: Vec<crate::utils::ModelResponse> = r.phase1_responses.iter()
                                        .map(|mr| crate::utils::ModelResponse {
                                            model_id: mr.model_id.clone(),
                                            content: mr.content.clone(),
                                            error_message: mr.error_message.clone(),
                                        })
                                        .collect();
                                    let final_consensus = r.phase3_consensus.as_ref().map(|c| c.content.clone());
                                    crate::utils::CollaborativeRound {
                                        user_message: r.user_question.clone(),
                                        model_responses,
                                        final_consensus,
                                    }
                                })
                                .collect();
                            let history = crate::utils::CollaborativeHistory {
                                rounds: history_rounds,
                                selected_models: selected_models_for_save.clone(),
                                system_prompt: String::new(),
                            };
                            let history_enum = ChatHistory::Collaborative(history.clone());
                            if ChatHistory::has_content(&history_enum) {
                                let summary = ChatHistory::generate_chat_summary(&history_enum);
                                let session = ChatSession {
                                    id: sid.clone(),
                                    title: summary,
                                    mode: ChatMode::Collaborative,
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
                        }
                    }
                    Err(e) => {
                        // Handle error
                        let _ = try_signal_update(&mut conversation_history_clone, |history| {
                            if let Some(last_round) = history.last_mut() {
                                last_round.phase1_responses = models
                                    .iter()
                                    .map(|id| ModelResponse {
                                        model_id: id.clone(),
                                        content: String::new(),
                                        error_message: Some(e.clone()),
                                    })
                                    .collect();
                            }
                        });
                        try_signal_set(&mut is_processing_clone, false);
                        
                        // Auto-save even on error (only when there is content; spawn_blocking to avoid blocking async runtime)
                        if let Some(sid) = session_id_for_save {
                            let history_rounds: Vec<crate::utils::CollaborativeRound> = try_signal_read(&conversation_history_clone, |history| history.clone())
                                .unwrap_or_default()
                                .iter()
                                .map(|r| {
                                    let model_responses: Vec<crate::utils::ModelResponse> = r.phase1_responses.iter()
                                        .map(|mr| crate::utils::ModelResponse {
                                            model_id: mr.model_id.clone(),
                                            content: mr.content.clone(),
                                            error_message: mr.error_message.clone(),
                                        })
                                        .collect();
                                    let final_consensus = r.phase3_consensus.as_ref().map(|c| c.content.clone());
                                    crate::utils::CollaborativeRound {
                                        user_message: r.user_question.clone(),
                                        model_responses,
                                        final_consensus,
                                    }
                                })
                                .collect();
                            let history = crate::utils::CollaborativeHistory {
                                rounds: history_rounds,
                                selected_models: selected_models_for_save.clone(),
                                system_prompt: String::new(),
                            };
                            let history_enum = ChatHistory::Collaborative(history.clone());
                            if ChatHistory::has_content(&history_enum) {
                                let summary = ChatHistory::generate_chat_summary(&history_enum);
                                let session = ChatSession {
                                    id: sid.clone(),
                                    title: summary,
                                    mode: ChatMode::Collaborative,
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
                        }
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
                ChatMode::Collaborative,
                "Collaborative round".to_string(),
                task,
                cancel_flag,
            );
        }
    };

    // Get filtered models for display
    let models_result = available_models.read();
    let loading = models_result.is_none();
    let (models_list, error) = match &*models_result {
        Some(Ok(models)) => (models.clone(), None),
        Some(Err(e)) => (Vec::new(), Some(e.clone())),
        None => (Vec::new(), None),
    };

    let filtered_models: Vec<Model> = {
        let search = search_query.read().to_lowercase();
        if search.is_empty() {
            models_list.clone()
        } else {
            models_list
                .iter()
                .filter(|m| {
                    m.display_name().to_lowercase().contains(&search)
                        || m.id.to_lowercase().contains(&search)
                })
                .cloned()
                .collect()
        }
    };

    rsx! {
        div {
            class: "flex flex-col h-full",

            // Model Selection Screen
            if *selection_step.read() == 0 {
                if let Some(_client_arc) = &client {
                    // Header
                    div {
                        class: "p-4 border-b border-[var(--color-base-300)]",
                        h2 {
                            class: "text-lg font-bold text-[var(--color-base-content)] mb-1",
                            "Select Models for Collaboration"
                        }
                        p {
                            class: "text-xs text-[var(--color-base-content)]/70",
                            "Choose 2 or more AI models that will work together to answer your questions."
                        }
                    }

                    // Search box
                    if !loading && error.is_none() {
                        div {
                            class: "px-4 pt-2",
                            input {
                                r#type: "text",
                                value: "{search_query}",
                                oninput: move |evt| search_query.set(evt.value().clone()),
                                placeholder: "Search models...",
                                class: "w-full px-3 py-2 text-sm rounded bg-[var(--color-base-100)] text-[var(--color-base-content)] border border-[var(--color-base-300)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent",
                            }
                        }
                    }

                    // Model list
                    div {
                        class: "flex-1 overflow-y-auto p-4",

                        if loading {
                            div {
                                class: "flex items-center justify-center h-full",
                                div {
                                    class: "text-center",
                                    div {
                                        class: "flex justify-center mb-4",
                                        img {
                                            src: asset!("/assets/loading.svg"),
                                            class: "w-16 h-16 animate-spin",
                                            alt: "Loading",
                                        }
                                    }
                                    p {
                                        class: "text-[var(--color-base-content)]/70",
                                        "Loading available models..."
                                    }
                                }
                            }
                        } else if let Some(err) = &error {
                            div {
                                class: "flex items-center justify-center h-full",
                                div {
                                    class: "text-center max-w-md",
                                    div {
                                        class: "flex justify-center mb-4",
                                        img {
                                            src: asset!("/assets/alert.svg"),
                                            class: "w-16 h-16",
                                            alt: "Error",
                                        }
                                    }
                                    p {
                                        class: "text-red-500 mb-2",
                                        "{err}"
                                    }
                                    p {
                                        class: "text-sm text-[var(--color-base-content)]/70",
                                        "Please check your API key in settings."
                                    }
                                }
                            }
                        } else {
                            div {
                                class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2",

                                for model in filtered_models.iter() {
                                    {
                                        let model_id = model.id.clone();
                                        let display_name = model.display_name();
                                        let is_selected = selected_models.read().contains(&model_id);

                                        rsx! {
                                            button {
                                                key: "{model_id}",
                                                onclick: move |_| toggle_model(model_id.clone()),
                                                class: if is_selected {
                                                    "p-3 rounded border-2 border-[var(--color-primary)] bg-[var(--color-primary)]/10 transition-all text-left"
                                                } else {
                                                    "p-3 rounded border border-[var(--color-base-300)] bg-[var(--color-base-200)] hover:border-[var(--color-primary)]/50 transition-all text-left"
                                                },
                                                div {
                                                    class: "flex items-start gap-2",
                                                    div {
                                                        class: "flex-shrink-0 mt-0.5",
                                                        if is_selected {
                                                            span { class: "text-[var(--color-primary)] text-sm", "✓" }
                                                        } else {
                                                            span { class: "text-[var(--color-base-content)]/30 text-sm", "○" }
                                                        }
                                                    }
                                                    div {
                                                        class: "flex-1 min-w-0",
                                                        div {
                                                            class: "font-semibold text-sm text-[var(--color-base-content)] truncate",
                                                            "{display_name}"
                                                        }
                                                        div {
                                                            class: "text-xs text-[var(--color-base-content)]/50 truncate",
                                                            "{model.id}"
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if filtered_models.is_empty() {
                                div {
                                    class: "text-center py-8",
                                    p {
                                        class: "text-[var(--color-base-content)]/70",
                                        "No models found."
                                    }
                                }
                            }
                        }
                    }

                    // Footer
                    if !loading && error.is_none() {
                        div {
                            class: "p-4 border-t border-[var(--color-base-300)]",
                            div {
                                class: "flex items-center justify-between gap-2",
                                div {
                                    class: "text-sm text-[var(--color-base-content)]/70",
                                    "{selected_models.read().len()} models selected (minimum 2)"
                                }
                                button {
                                    onclick: start_chat,
                                    disabled: selected_models.read().len() < 2,
                                    class: "px-4 py-2 text-sm rounded bg-[var(--color-primary)] text-[var(--color-primary-content)] hover:bg-[var(--color-primary)]/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-all",
                                    "Start Collaborative Chat"
                                }
                            }
                        }
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
                // Chat Interface

                // Prompt Templates Section
                div {
                    class: "p-3 border-b border-[var(--color-base-300)] bg-[var(--color-base-100)]",

                    div {
                        class: "flex items-center justify-between mb-2",
                        h3 {
                            class: "text-sm font-semibold text-[var(--color-base-content)]",
                            "Prompt Templates (Click to customize)"
                        }
                        button {
                            onclick: move |_| {
                                selection_step.set(0);
                                conversation_history.write().clear();
                            },
                            class: "text-xs text-[var(--color-primary)] hover:underline",
                            "Change Models"
                        }
                    }

                    div {
                        class: "grid grid-cols-1 md:grid-cols-3 gap-2",

                        PromptCard {
                            theme,
                            title: "Initial Response".to_string(),
                            phase_number: 1,
                            prompt: prompt_templates.read().initial_response.clone(),
                            on_edit: move |_| open_prompt_editor(PromptType::Initial),
                        }

                        PromptCard {
                            theme,
                            title: "Review Feedback".to_string(),
                            phase_number: 2,
                            prompt: prompt_templates.read().cross_review.clone(),
                            on_edit: move |_| open_prompt_editor(PromptType::Review),
                        }

                        PromptCard {
                            theme,
                            title: "Consensus Synthesis".to_string(),
                            phase_number: 3,
                            prompt: prompt_templates.read().consensus.clone(),
                            on_edit: move |_| open_prompt_editor(PromptType::Consensus),
                        }
                    }
                }

                // Chat area
                div {
                    class: "flex-1 min-h-0 overflow-y-auto p-4",

                    if conversation_history.read().is_empty() {
                        // Empty state
                        div {
                            class: "flex flex-col items-center justify-center h-full",
                            h2 {
                                class: "text-lg sm:text-xl md:text-2xl font-bold text-[var(--color-base-content)] mb-2",
                                "🤝 Collaborative Mode Ready"
                            }
                            p {
                                class: "text-sm sm:text-base text-[var(--color-base-content)]/70 mb-4 text-center max-w-md",
                                "Selected models will work together through three phases: initial responses, cross-review, and consensus synthesis."
                            }
                            div {
                                class: "text-xs text-[var(--color-base-content)]/60 space-y-1",
                                for (idx, model_id) in selected_models.read().iter().enumerate() {
                                    p { key: "{idx}", "• {model_id}" }
                                }
                            }
                        }
                    } else {
                        // Conversation display
                        div {
                            class: "space-y-8 w-full",

                            for (round_idx, round) in conversation_history.read().iter().enumerate() {
                                div {
                                    key: "{round_idx}",

                                    // User message
                                    div {
                                        class: "flex justify-end mb-4",
                                        div {
                                            class: "max-w-[85%] bg-[var(--color-primary)] text-[var(--color-primary-content)] px-3 sm:px-4 md:px-5 py-2 sm:py-3 rounded-lg text-sm sm:text-base",
                                            FormattedText {
                                                theme,
                                                content: round.user_question.clone(),
                                            }
                                        }
                                    }

                                    // Phase 1: Initial Responses
                                    if !round.phase1_responses.is_empty() {
                                        div {
                                            class: "mb-6",

                                            // Phase header
                                            div {
                                                class: "flex items-center gap-2 mb-3",
                                                span {
                                                    class: "inline-block px-2 py-1 rounded text-xs font-semibold text-white {CollaborativePhase::Initial.badge_color()}",
                                                    "{CollaborativePhase::Initial.name()}"
                                                }
                                            }

                                            // Responses grid
                                            div {
                                                class: "grid grid-cols-1 md:grid-cols-2 gap-3",

                                                for response in round.phase1_responses.iter() {
                                                    div {
                                                        key: "{response.model_id}",
                                                        class: if response.error_message.is_some() {
                                                            "bg-red-500/10 rounded-lg p-4 border-2 border-red-500/50"
                                                        } else {
                                                            "bg-[var(--color-base-200)] rounded-lg p-4 border border-[var(--color-base-300)]"
                                                        },

                                                        div {
                                                            class: "text-sm font-bold text-[var(--color-base-content)] mb-2 truncate",
                                                            "{response.model_id}"
                                                        }

                                                        if let Some(error) = &response.error_message {
                                                            div {
                                                                class: "text-sm text-red-500",
                                                                "Error: {error}"
                                                            }
                                                        } else {
                                                            div {
                                                                class: "text-sm text-[var(--color-base-content)]",
                                                                FormattedText {
                                                                    theme,
                                                                    content: response.content.clone(),
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Phase 2: Cross-Review
                                    if !round.phase2_reviews.is_empty() {
                                        div {
                                            class: "mb-6",

                                            div {
                                                class: "flex items-center gap-2 mb-3",
                                                span {
                                                    class: "inline-block px-2 py-1 rounded text-xs font-semibold text-white {CollaborativePhase::Review.badge_color()}",
                                                    "{CollaborativePhase::Review.name()}"
                                                }
                                            }

                                            div {
                                                class: "grid grid-cols-1 md:grid-cols-2 gap-3",

                                                for review in round.phase2_reviews.iter() {
                                                    div {
                                                        key: "{review.model_id}",
                                                        class: if review.error_message.is_some() {
                                                            "bg-red-500/10 rounded-lg p-4 border-2 border-red-500/50"
                                                        } else {
                                                            "bg-[var(--color-base-200)] rounded-lg p-4 border border-[var(--color-base-300)]"
                                                        },

                                                        div {
                                                            class: "text-sm font-bold text-[var(--color-base-content)] mb-2 truncate",
                                                            "{review.model_id}'s Review"
                                                        }

                                                        if let Some(error) = &review.error_message {
                                                            div {
                                                                class: "text-sm text-red-500",
                                                                "Error: {error}"
                                                            }
                                                        } else {
                                                            div {
                                                                class: "text-sm text-[var(--color-base-content)]",
                                                                FormattedText {
                                                                    theme,
                                                                    content: review.content.clone(),
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Phase 3: Consensus
                                    if let Some(consensus) = &round.phase3_consensus {
                                        div {
                                            class: "mb-6",

                                            div {
                                                class: "flex items-center gap-2 mb-3",
                                                span {
                                                    class: "inline-block px-2 py-1 rounded text-xs font-semibold text-white {CollaborativePhase::Consensus.badge_color()}",
                                                    "{CollaborativePhase::Consensus.name()}"
                                                }
                                            }

                                            div {
                                                class: if consensus.error_message.is_some() {
                                                    "bg-red-500/10 rounded-lg p-4 border-2 border-red-500/50"
                                                } else {
                                                    "bg-green-500/10 rounded-lg p-4 border-2 border-green-500/50"
                                                },

                                                div {
                                                    class: "text-sm font-bold text-[var(--color-base-content)] mb-2",
                                                    "🎯 Collaborative Answer (synthesized by {consensus.model_id})"
                                                }

                                                if let Some(error) = &consensus.error_message {
                                                    div {
                                                        class: "text-sm text-red-500",
                                                        "Error: {error}"
                                                    }
                                                } else {
                                                    div {
                                                        class: "text-sm text-[var(--color-base-content)]",
                                                        FormattedText {
                                                            theme,
                                                            content: consensus.content.clone(),
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Streaming indicators
                            if *is_processing.read() && !current_streaming_responses.read().is_empty() {
                                div {
                                    class: "mb-6",

                                    div {
                                        class: "flex items-center gap-2 mb-3",
                                        span {
                                            class: "inline-block px-2 py-1 rounded text-xs font-semibold text-white {current_phase.read().badge_color()}",
                                            "{current_phase.read().name()}"
                                        }
                                        span {
                                            class: "inline-block w-2 h-2 bg-[var(--color-primary)] rounded-full animate-pulse"
                                        }
                                    }

                                    div {
                                        class: if *current_phase.read() == CollaborativePhase::Consensus {
                                            "bg-green-500/10 rounded-lg p-4 border-2 border-green-500/50"
                                        } else {
                                            "grid grid-cols-1 md:grid-cols-2 gap-3"
                                        },

                                        for (model_id, content) in current_streaming_responses.read().iter() {
                                            div {
                                                key: "{model_id}",
                                                class: if *current_phase.read() != CollaborativePhase::Consensus {
                                                    "bg-[var(--color-base-200)] rounded-lg p-4 border border-[var(--color-base-300)]"
                                                } else {
                                                    ""
                                                },

                                                div {
                                                    class: "text-sm font-bold text-[var(--color-base-content)] mb-2 flex items-center gap-2",
                                                    if model_id == "consensus" {
                                                        span { "🎯 Synthesizing Collaborative Answer..." }
                                                    } else {
                                                        span { "{model_id}" }
                                                    }
                                                    span {
                                                        class: "inline-block w-2 h-2 bg-[var(--color-primary)] rounded-full animate-pulse"
                                                    }
                                                }

                                                div {
                                                    class: "text-sm text-[var(--color-base-content)] min-h-[3rem]",
                                                    div {
                                                        class: "whitespace-pre-wrap break-words",
                                                        "{content}"
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
                                "Cancelling active collaborative run..."
                            } else {
                                "This collaborative run is active in the background. You can cancel it manually."
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
                            if is_cancelling { "Cancelling..." } else { "Cancel Run" }
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

            // Prompt Editor Modal
            PromptEditorModal {
                theme,
                open: prompt_editor_open,
                prompt_type: *editing_prompt_type.read(),
                current_prompt: temp_prompt(),
                default_prompt: PromptTemplates::default().get(*editing_prompt_type.read()),
                on_save: save_prompt,
            }
        }
    }
}
