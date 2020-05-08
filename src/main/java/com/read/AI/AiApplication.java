package com.read.AI;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@SpringBootApplication
public class AiApplication {

	@RequestMapping("/")
	public String test() {
		return "Hello world";
	}

	public static void main(String[] args) {
		SpringApplication.run(AiApplication.class, args);
	}

}
