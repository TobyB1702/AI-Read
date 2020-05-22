package com.read.AI.controller;

import com.read.AI.service.AIService;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
public class AIController {
    AIService AIService = new AIService();

    @RequestMapping("/")
    public String InitilazeModel() throws IOException {
        return AIService.CreateModel();
    }

}
