package com.read.AI.controller;

import com.read.AI.service.AIService;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class AIController {
    AIService AIService = new AIService();

    @RequestMapping("/")
    public String test() {
        return AIService.getMessage();
    }

}
