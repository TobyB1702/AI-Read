package com.read.AI.controller;

import com.read.AI.service.AIService;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
public class AIController {
    private static final AIService AIService = new AIService();
    private static final String FILE_PATH = "AI-Read-Model.zip";


    @RequestMapping("/")
    public String InitilazeModel() throws IOException {
        AIService.DoesModelFileExist(FILE_PATH);
        AIService.TestExampleImage();
        return "Model Created";
    }
}
