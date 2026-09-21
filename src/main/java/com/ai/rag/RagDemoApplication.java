package com.ai.rag;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

/**
 * Spring Boot entry point for the RAG demo.
 *
 * <p>Stack: Spring Boot 3 (Web MVC + SSE), langchain4j 0.34.0 (models, RAG,
 * embedding store) and a local Ollama server running llama3.2:1b.
 */
@SpringBootApplication
@ConfigurationPropertiesScan
public class RagDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(RagDemoApplication.class, args);
    }
}
