package com.read.AI.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class testController {


    @RequestMapping("/")
    public String test() {
        return "BackEnd Message";
    }

}
