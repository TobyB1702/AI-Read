FROM openjdk:8-jdk-alpine
ADD target/ app.jar
ENTRYPOINT ["java","-jar","app.jar"]
EXPOSE 8080