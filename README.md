Vernaim — Real-Time Facial Recognition & Access Control

**A Python-based computer vision system for facial identification, authentication, and controlled access, with runtime recovery and security mechanisms.**

**Project status:** Completed · Final version: V3

 Overview

Vernaim is a real-time facial recognition and access control application developed to combine computer vision with authentication, session protection, and operational reliability.

Beyond facial recognition itself, the project addresses the challenges of operating an access control system continuously: handling camera failures, managing authentication attempts, protecting restricted routes, and maintaining predictable behavior during extended runtime.

The final version, V3, integrates the recognition workflow with application-level security mechanisms and recovery procedures. It was subjected to extended runtime testing, including overnight operation and practical tests involving authentication, recognition states, camera recovery, and access from another device.

This repository documents the final implementation and its observed behavior. The project is complete and is not under active development.

Key Features

Computer Vision and Recognition

- Real-time facial recognition workflow.
- Identity verification and classification states.
- Visual feedback for recognition and verification status.
- Integration with a camera-based video stream.
- Recognition status handling as part of the access control workflow.

Authentication and Access Control

- Password-based authentication.
- Password hashing using PBKDF2-HMAC-SHA256 with 600,000 iterations.
- Temporary lockout after repeated failed authentication attempts.
- Configured threshold of five failed attempts within a five-minute window, followed by a five-minute lockout.
- Protected application routes and access restrictions.
- Session-related security controls.

Runtime Reliability

- Camera stream monitoring.
- Recovery attempts after camera disconnection.
- Detection and handling of stale video streams.
- Continued operation under extended runtime conditions.
- Atomic file-writing mechanisms and backup-related handling for application data.

 Web Application Security

- CSRF protection for state-changing requests.
- Security-related HTTP response headers.
- Authentication barriers for protected routes.
- Failed-login handling and temporary lockout.

These mechanisms provide multiple layers of application-level protection. They should not be interpreted as proof that the application is immune to attacks.

System Behavior

The application combines several stages into a single workflow:

1. **Application startup:** Initialize the application and its required components.
2. **Authentication:** Require valid credentials before granting access to protected functionality.
3. **Video acquisition:** Obtain frames from the connected camera and monitor stream availability.
4. **Recognition and classification:** Process the available visual input and determine the corresponding recognition state.
5. **Status feedback:** Present visual feedback, including a yellow state during the observed verification/classification workflow.
6. **Access handling:** Apply the application's authentication and access-control logic.
7. **Failure recovery:** Handle camera interruptions and attempt to restore the video stream.
8. **Security enforcement:** Apply authentication restrictions, failed-attempt lockout, and relevant request protections.

The exact behavior of individual recognition states depends on the implementation and its configured verification workflow.

Reliability and Security Design

Vernaim incorporates security controls and recovery mechanisms designed to improve access protection and operational reliability.

- **Authentication and authorization:** Password hashing, protected routes, and temporary lockout after repeated authentication failures.
- **Web security:** CSRF tokens for state-changing requests and security-related HTTP response headers.
- **Stream reliability:** Camera interruption handling, reconnection attempts, and stale-stream detection.
- **Data persistence:** Atomic file writes and backup-related mechanisms intended to reduce the risk of data corruption.

The authentication lockout uses an in-memory, IP-based mechanism. Its behavior depends on the application's runtime and deployment configuration; it is not designed as a distributed rate-limiting system.

Testing and Validation

The final V3 implementation underwent extended manual runtime testing, including operation throughout the night or longer.

The application was evaluated through practical test scenarios to observe behavior under specific conditions.

> **Testing Scope Notice:** These tests provide practical evidence under the conditions used, but are not equivalent to exhaustive automated testing, formal verification, or an independent penetration test. No universal performance claims are made because standardized measurements for latency, frame rate, throughput, and resource consumption are outside the scope of this documentation.

Core Functionality and User Flow

- **Startup and Authentication:** Observation of application startup and authentication workflows.
- **Recognition-State Feedback:** Observation of system indicators, including the yellow state, and subsequent classification or verification behavior.
- **Multi-Device Access:** Verification of remote accessibility from a separate device under the tested setup.

Resilience and Fault Tolerance

- **Camera Disruption:** Observation of disconnection handling and stream recovery.
- **Repeated Failed Logins:** Observation of the temporary IP-based lockout after repeated authentication failures.
- **Extended Runtime:** Monitoring of general application stability and behavior during prolonged operation.

limitations and Security Considerations

Facial recognition and access control require careful consideration of security, reliability, and privacy.

- **Presentation attacks:** The implemented liveness workflow uses head-movement checks, but this mechanism does not guarantee protection against photographs, video replays, or other presentation attacks.
- **Liveness configuration:** Verification behavior depends on the configured mode. A configuration that disables liveness checks must not be interpreted as providing equivalent verification guarantees.
- **Authentication lockout:** The in-memory, IP-based mechanism has limitations compared with persistent, centralized rate limiting.
- **Security assurance:** The documented runtime tests do not establish resistance to every attack or deployment failure.
- **Biometric privacy:** Facial images and recognition data must be handled responsibly and protected against unauthorized access.
- **Deployment security:** Production use requires appropriate secret management, access restrictions, secure transport, and deployment-specific review.

Vernaim should be evaluated according to its actual configuration and deployment environment before being used for security-critical access decisions.

Technology

The project uses a Python-based computer vision and web application stack, including the following technologies and concepts where applicable in the implementation:

- **Python** — application logic and system integration.
- **OpenCV** — computer vision and camera/video processing.
- **Deep learning and facial recognition** — visual identification workflow.
- **Web application components** — authentication and access through a browser-based interface.
- **Cryptographic password hashing** — PBKDF2-HMAC-SHA256.
- **Session and request security** — access restrictions, CSRF protection, and HTTP security headers.

The complete dependency list and environment-specific configuration should be taken from the actual project files rather than inferred from this overview.

Repository Structure

The final implementation is identified as:

`vernaim version final.py`

An earlier version is identified as:

`vernaimeasyaccess.py`

V3 is the final version documented by this repository. Earlier versions should be clearly labeled as historical implementations if included, so that the final entry point remains unambiguous.

Running the Project

Execution depends on the project's actual Python dependencies, camera configuration, authentication setup, and local environment.

Before running the application:

1. Review the source code and required dependencies.
2. Configure the required local settings and credentials.
3. Connect and configure the supported camera.
4. Install the dependencies required by the implementation.
5. Start the final V3 entry point using the appropriate Python environment.

Exact installation commands and configuration variables should be documented from the verified project environment.

**Security note:** Never commit passwords, secret keys, private configuration files, biometric databases, or personal facial images to the public repository.

Final Status

**Vernaim V3 is a completed project.**

The final implementation was exercised in extended runtime conditions and tested against the principal operational scenarios documented above. This repository serves as the project's final source-code and technical documentation record.

No further feature development is planned.

---

## Author

Developed as a computer vision and software engineering project focused on facial recognition, access control, application security, and runtime reliability.
