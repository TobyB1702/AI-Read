package com.read.AI.controller;

import com.read.AI.service.TestService;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController {
    TestService testService = new TestService();

    @RequestMapping("/")
    public String test() {
        return testService.getMessage();
    }

}
