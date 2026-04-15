use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

// ============================================================================
// Constants
// ============================================================================

const OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";
const APP_NAME: &str = "gtllm";
const APP_URL: &str = "https://github.com/yourusername/gtllm"; // Update with your repo URL

// ============================================================================
// API Types - Request
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "user", "assistant", "system"
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }
}

// ============================================================================
// API Types - Response
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    pub model: String,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub error: Option<ApiError>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: u32,
    #[serde(default)]
    pub message: Option<ResponseMessage>,
    #[serde(default)]
    pub delta: Option<Delta>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiError {
    pub code: u32,
    pub message: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

// ============================================================================
// Model Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub pricing: Option<ModelPricing>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub architecture: Option<ModelArchitecture>,
    #[serde(default)]
    pub top_provider: Option<TopProvider>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelPricing {
    pub prompt: String,
    pub completion: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelArchitecture {
    #[serde(default)]
    pub modality: Option<String>,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub instruct_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TopProvider {
    pub max_completion_tokens: Option<u32>,
    pub is_moderated: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<Model>,
}

// ============================================================================
// Credits Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreditsData {
    pub total_credits: f64,
    pub total_usage: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreditsResponse {
    pub data: CreditsData,
}

impl CreditsData {
    /// Get remaining credits (total_credits - total_usage)
    pub fn remaining(&self) -> f64 {
        self.total_credits - self.total_usage
    }

    /// Format remaining credits as a string with 2 decimal places
    pub fn remaining_formatted(&self) -> String {
        format!("${:.2}", self.remaining())
    }
}

// ============================================================================
// Stream Event Types
// ============================================================================

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Content(String),
    Done,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ModelStreamEvent {
    pub model_id: String,
    pub event: StreamEvent,
}

// ============================================================================
// OpenRouter Client
// ============================================================================

#[derive(Clone)]
pub struct OpenRouterClient {
    client: Client,
    api_key: Arc<String>,
    concurrency_limit: Arc<Semaphore>,
}

struct LimitedStream {
    inner: Pin<Box<dyn Stream<Item = StreamEvent> + Send>>,
    _permit: OwnedSemaphorePermit,
}

impl Stream for LimitedStream {
    type Item = StreamEvent;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl PartialEq for OpenRouterClient {
    fn eq(&self, other: &Self) -> bool {
        self.api_key == other.api_key
    }
}

impl OpenRouterClient {
    pub fn new(api_key: String) -> Result<Self, String> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .pool_max_idle_per_host(10) // Allow multiple concurrent connections per host
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        Ok(Self {
            client,
            api_key: Arc::new(api_key),
            concurrency_limit: Arc::new(Semaphore::new(4)),
        })
    }

    // ========================================================================
    // Fetch Available Models
    // ========================================================================

    pub async fn fetch_models(&self) -> Result<Vec<Model>, String> {
        let url = format!("{}/models", OPENROUTER_API_BASE);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", APP_URL)
            .header("X-Title", APP_NAME)
            .send()
            .await
            .map_err(|e| format!("Failed to fetch models: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as JSON error response
            let error_message = if let Ok(error_response) = serde_json::from_str::<serde_json::Value>(&error_text) {
                if let Some(error_obj) = error_response.get("error") {
                    if let Some(message) = error_obj.get("message").and_then(|m| m.as_str()) {
                        message.to_string()
                    } else {
                        error_text
                    }
                } else {
                    error_text
                }
            } else {
                error_text
            };

            return Err(format!("OpenRouter error ({}): {}", status, error_message));
        }

        let models_response: ModelsResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse models response: {}", e))?;

        Ok(models_response.data)
    }

    // ========================================================================
    // Fetch Credits
    // ========================================================================

    pub async fn fetch_credits(&self) -> Result<CreditsData, String> {
        let url = format!("{}/credits", OPENROUTER_API_BASE);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", APP_URL)
            .header("X-Title", APP_NAME)
            .send()
            .await
            .map_err(|e| format!("Failed to fetch credits: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as JSON error response
            let error_message = if let Ok(error_response) = serde_json::from_str::<serde_json::Value>(&error_text) {
                if let Some(error_obj) = error_response.get("error") {
                    if let Some(message) = error_obj.get("message").and_then(|m| m.as_str()) {
                        message.to_string()
                    } else {
                        error_text
                    }
                } else {
                    error_text
                }
            } else {
                error_text
            };

            return Err(format!("OpenRouter error ({}): {}", status, error_message));
        }

        let credits_response: CreditsResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse credits response: {}", e))?;

        Ok(credits_response.data)
    }

    // ========================================================================
    // Single Model Streaming Chat Completion
    // ========================================================================

    pub async fn stream_chat_completion(
        &self,
        model_id: String,
        messages: Vec<ChatMessage>,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, String> {
        let permit = self
            .concurrency_limit
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| format!("Failed to acquire concurrency permit: {}", e))?;

        let request = ChatCompletionRequest {
            model: model_id,
            messages,
            stream: Some(true),
            max_tokens: None,
            temperature: None,
            top_p: None,
        };

        let url = format!("{}/chat/completions", OPENROUTER_API_BASE);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", APP_URL)
            .header("X-Title", APP_NAME)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as JSON error response
            let error_message = if let Ok(error_response) = serde_json::from_str::<serde_json::Value>(&error_text) {
                if let Some(error_obj) = error_response.get("error") {
                    if let Some(message) = error_obj.get("message").and_then(|m| m.as_str()) {
                        message.to_string()
                    } else {
                        error_text
                    }
                } else {
                    error_text
                }
            } else {
                error_text
            };

            return Err(format!("OpenRouter error ({}): {}", status, error_message));
        }

        // Parse SSE safely across arbitrary network chunk boundaries.
        let stream = futures::stream::unfold(
            (response.bytes_stream(), String::new(), VecDeque::<StreamEvent>::new(), false),
            |(mut bytes_stream, mut partial, mut pending, mut finished)| async move {
                loop {
                    if let Some(event) = pending.pop_front() {
                        return Some((event, (bytes_stream, partial, pending, finished)));
                    }

                    if finished {
                        return None;
                    }

                    match bytes_stream.next().await {
                        Some(Ok(bytes)) => {
                            partial.push_str(&String::from_utf8_lossy(&bytes));
                            for event in parse_sse_from_buffer(&mut partial, false) {
                                pending.push_back(event);
                            }
                        }
                        Some(Err(e)) => {
                            pending.push_back(StreamEvent::Error(format!("Stream error: {}", e)));
                            finished = true;
                        }
                        None => {
                            for event in parse_sse_from_buffer(&mut partial, true) {
                                pending.push_back(event);
                            }
                            finished = true;
                        }
                    }
                }
            },
        );

        Ok(Box::pin(LimitedStream {
            inner: Box::pin(stream),
            _permit: permit,
        }))
    }

    // ========================================================================
    // Multiple Models Concurrent Streaming
    // ========================================================================

    pub async fn stream_chat_completion_multi(
        &self,
        model_ids: Vec<String>,
        messages: Vec<ChatMessage>,
    ) -> Result<mpsc::UnboundedReceiver<ModelStreamEvent>, String> {
        let (tx, rx) = mpsc::unbounded_channel();

        for model_id in model_ids {
            let client = self.clone();
            let messages = messages.clone();
            let tx = tx.clone();
            let model_id_clone = model_id.clone();

            tokio::spawn(async move {
                match client.stream_chat_completion(model_id.clone(), messages).await {
                    Ok(mut stream) => {
                        while let Some(event) = stream.next().await {
                            let model_event = ModelStreamEvent {
                                model_id: model_id_clone.clone(),
                                event: event.clone(),
                            };

                            if tx.send(model_event).is_err() {
                                // Receiver dropped, stop streaming
                                break;
                            }

                            // If we hit Done or Error, stop this stream
                            if matches!(event, StreamEvent::Done | StreamEvent::Error(_)) {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(ModelStreamEvent {
                            model_id: model_id_clone,
                            event: StreamEvent::Error(e),
                        });
                    }
                }
            });
        }

        Ok(rx)
    }

    // ========================================================================
    // Non-Streaming Chat Completion
    // ========================================================================

    pub async fn chat_completion(
        &self,
        model_id: String,
        messages: Vec<ChatMessage>,
    ) -> Result<ChatCompletionResponse, String> {
        let request = ChatCompletionRequest {
            model: model_id,
            messages,
            stream: Some(false),
            max_tokens: None,
            temperature: None,
            top_p: None,
        };

        let url = format!("{}/chat/completions", OPENROUTER_API_BASE);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", APP_URL)
            .header("X-Title", APP_NAME)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as JSON error response
            let error_message = if let Ok(error_response) = serde_json::from_str::<serde_json::Value>(&error_text) {
                if let Some(error_obj) = error_response.get("error") {
                    if let Some(message) = error_obj.get("message").and_then(|m| m.as_str()) {
                        message.to_string()
                    } else {
                        error_text
                    }
                } else {
                    error_text
                }
            } else {
                error_text
            };

            return Err(format!("OpenRouter error ({}): {}", status, error_message));
        }

        let completion_response: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        // Check for errors in the response
        if let Some(error) = &completion_response.error {
            return Err(format!("API error: {}", error.message));
        }

        Ok(completion_response)
    }
}

// ============================================================================
// SSE Parsing Helper
// ============================================================================

fn parse_sse_from_buffer(buffer: &mut String, flush_remaining: bool) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    while let Some(newline_idx) = buffer.find('\n') {
        let mut line: String = buffer.drain(..=newline_idx).collect();
        if line.ends_with('\n') {
            line.pop();
        }
        if line.ends_with('\r') {
            line.pop();
        }
        parse_sse_line(&line, &mut events);
    }

    if flush_remaining && !buffer.trim().is_empty() {
        let line = buffer.trim().to_string();
        parse_sse_line(&line, &mut events);
        buffer.clear();
    }

    events
}

fn parse_sse_line(line: &str, events: &mut Vec<StreamEvent>) {
    let line = line.trim();

    // Skip empty lines and comments
    if line.is_empty() || line.starts_with(':') {
        return;
    }

    if let Some(data) = line.strip_prefix("data: ") {
        if data == "[DONE]" {
            events.push(StreamEvent::Done);
            return;
        }

        match serde_json::from_str::<ChatCompletionResponse>(data) {
            Ok(response) => {
                if let Some(error) = response.error {
                    events.push(StreamEvent::Error(error.message));
                    return;
                }

                if let Some(choice) = response.choices.first() {
                    if let Some(delta) = &choice.delta {
                        if let Some(content) = &delta.content {
                            if !content.is_empty() {
                                events.push(StreamEvent::Content(content.clone()));
                            }
                        }
                    }

                    if let Some(finish_reason) = &choice.finish_reason {
                        if finish_reason == "error" {
                            events.push(StreamEvent::Error(
                                "Stream terminated with error".to_string(),
                            ));
                        } else if !finish_reason.is_empty() {
                            events.push(StreamEvent::Done);
                        }
                    }
                }
            }
            Err(e) => {
                // Do not log the raw payload to avoid leaking prompt/response contents.
                eprintln!(
                    "Failed to parse SSE chunk: {} (payload_len={})",
                    e,
                    data.len()
                );
            }
        }
    }
}

fn parse_sse_chunk(text: &str) -> Vec<StreamEvent> {
    let mut buffer = text.to_string();
    let mut events = Vec::new();
    events.extend(parse_sse_from_buffer(&mut buffer, true));
    events
}

// ============================================================================
// Helper Functions
// ============================================================================

impl Model {
    /// Get a human-readable display name for the model
    pub fn display_name(&self) -> String {
        if !self.name.is_empty() {
            self.name.clone()
        } else {
            self.id.clone()
        }
    }

    /// Get pricing information as a formatted string
    pub fn pricing_info(&self) -> Option<String> {
        self.pricing.as_ref().map(|p| {
            format!(
                "Prompt: ${}/token, Completion: ${}/token",
                p.prompt, p.completion
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_constructors() {
        let user_msg = ChatMessage::user("Hello");
        assert_eq!(user_msg.role, "user");
        assert_eq!(user_msg.content, "Hello");

        let assistant_msg = ChatMessage::assistant("Hi there");
        assert_eq!(assistant_msg.role, "assistant");

        let system_msg = ChatMessage::system("You are helpful");
        assert_eq!(system_msg.role, "system");
    }

    #[test]
    fn test_parse_sse_done() {
        let chunk = "data: [DONE]\n";
        let events = parse_sse_chunk(chunk);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Done));
    }

    #[test]
    fn test_parse_sse_comment() {
        let chunk = ": OPENROUTER PROCESSING\n";
        let events = parse_sse_chunk(chunk);
        assert_eq!(events.len(), 0); // Comments should be ignored
    }

    #[test]
    fn test_parse_sse_buffered_split_chunks() {
        let mut buffer = String::new();
        buffer.push_str("data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hel");
        let events = parse_sse_from_buffer(&mut buffer, false);
        assert!(events.is_empty());

        buffer.push_str("lo\"},\"finish_reason\":null}],\"created\":1,\"model\":\"m\"}\n");
        let events = parse_sse_from_buffer(&mut buffer, false);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Content(_)));
    }
}
