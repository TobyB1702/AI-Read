FROM openjdk:8-jdk-alpine
ADD build/ app.jar
ENTRYPOINT ["java","-jar","app.jar"]
EXPOSE 8080