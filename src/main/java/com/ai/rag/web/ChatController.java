package com.ai.rag.web;

import com.ai.rag.config.RagProperties;
import com.ai.rag.service.KnowledgeBaseService;
import com.ai.rag.service.RagService;
import com.ai.rag.service.RagService.SourceInfo;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.List;
import java.util.Map;

/**
 * JSON + SSE API of the demo.
 *
 * <pre>
 *   GET  /api/health                  - service, model and corpus status
 *   POST /api/retrieve                - {"question": "..."} -> sources with scores
 *   GET  /api/chat/stream?question=..  - SSE: sources / token / done / error events
 * </pre>
 *
 * The chat endpoint is a GET because native {@code EventSource} clients can
 * only issue GET requests - the standard way to consume SSE from a browser.
 */
@RestController
@RequestMapping("/api")
public class ChatController {

    private final RagService ragService;
    private final KnowledgeBaseService knowledgeBase;
    private final RagProperties props;

    public ChatController(RagService ragService, KnowledgeBaseService knowledgeBase, RagProperties props) {
        this.ragService = ragService;
        this.knowledgeBase = knowledgeBase;
        this.props = props;
    }

    // --- Types ---------------------------------------------------------------

    public record HealthStatus(String status, OllamaInfo ollama, String embedding, KbInfo knowledgeBase) {
    }

    public record OllamaInfo(String url, String model, boolean up) {
    }

    public record KbInfo(int files, int chunks) {
    }

    public record QuestionRequest(String question) {
    }

    // --- Endpoints -------------------------------------------------------------

    @GetMapping("/health")
    public HealthStatus health() {
        return new HealthStatus(
                "ok",
                new OllamaInfo(props.ollama().baseUrl(), props.ollama().model(), knowledgeBase.ollamaUp()),
                knowledgeBase.embeddingDescription(),
                new KbInfo(knowledgeBase.fileCount(), knowledgeBase.segmentCount()));
    }

    @PostMapping("/retrieve")
    public Map<String, Object> retrieve(@RequestBody QuestionRequest request) {
        String question = validated(request == null ? null : request.question());
        List<SourceInfo> sources = ragService.retrieveSources(question);
        return Map.of("question", question, "sources", sources);
    }

    @GetMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamChat(@RequestParam String question) {
        return ragService.streamChat(validated(question));
    }

    // --- Helpers ----------------------------------------------------------------

    private static String validated(String question) {
        if (question == null || question.isBlank()) {
            throw new InvalidQuestionException("question must not be empty");
        }
        return question.strip();
    }

    /** Maps to a clean 400 instead of a stack trace. */
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    static class InvalidQuestionException extends RuntimeException {
        InvalidQuestionException(String message) {
            super(message);
        }
    }
}
