DEMETER PROJECT: Architectural Philosophy & Logical Workflow
This workflow document outlines the guiding principles, conceptual architecture, and logical constraints of the Demeter project. It is designed to instruct an AI agent or development team on the reasoning behind the system, ensuring that any code written aligns with the project's core ethos of democratizing precision agriculture.

1. Project Ethos: The Accessibility Gap
The fundamental philosophy driving Demeter is the democratization of high-fidelity agricultural data. Industrial precision agriculture relies on prohibitively expensive thermal imaging and live-feed sensor arrays. Small-scale farmers and hobbyists often overcompensate with harmful pesticides or water waste because they lack diagnostic tools.

The Demeter Mandate:
Replace heavy hardware requirements with intelligent software. The system must run efficiently on edge devices (like a Raspberry Pi or standard laptop) or via a lightweight web interface, translating complex data into immediate, actionable insights for non-expert users.

2. The Dual-Stream Epistemology
Plant health cannot be understood in a vacuum. A visual symptom is the result of a condition; the environment is the catalyst. Demeter’s logic relies on synthesizing these two realities through a decoupled, dual-model architecture.

Stream A: The Visual Reality (Current State)
Logical Purpose: To diagnose existing biological trauma or disease.

The Mechanism: Transfer learning via a MobileNetV2 Convolutional Neural Network (CNN).

The Rationale: We use transfer learning not just for computational efficiency, but because learning from 300,000+ images (PlantVillage/PlantNet) allows a lightweight edge model to possess the diagnostic pattern recognition of an industrial system. It interprets the now.

Stream B: The Environmental Reality (Future Trajectory)
Logical Purpose: To model the continuous relationship between environmental inputs (hydration, temperature) and physical growth (biomass).

The Mechanism: A Random Forest Regressor.

The Rationale: Unlike a classifier that groups data into rigid categories, a regressor understands the fluidity of nature. By isolating core features like temporal water use and pot weight, the model predicts the physical trajectory of the plant, anticipating stress before it visually manifests.

3. The Logic of Action: The Status Engine & Model Attribution
A core philosophical pillar of Demeter is Actionable Intelligence. Raw machine learning outputs (e.g., "78% probability of Alternaria solani") are useless to a gardener without context.

Source Attribution Mandate: Project-wide rule: wherever a decision is made or information is provided to the user, the system MUST clearly note which specific ML model it came from. Actionable intelligence should preferably be derived directly from model outputs (e.g., a Random Forest predicting specific water needs).

Disclosed Heuristics: While models are preferred, the system may feature a deterministic logic layer—the Status Engine—that acts as a translator between statistical probability and human intervention. However, any use of arbitrary heuristic rules or hardcoded thresholds MUST be unequivocally disclosed to the user (e.g., using `[HEURISTIC]` tags), so they are never confused with biological ground truth.

Abstraction of Complexity: The user interface should hide the underlying tensors and raw tabular data, but it must never hide the source of its intelligence. The user should see the synthesis: What is wrong, what will happen, what they must do, and exactly which model told them so.

4. Evaluation Philosophy: Measuring What Matters
The logic of how we define "success" must align with the physical realities of agriculture.

Why F1-Score over Accuracy? In nature, healthy plants are common, and specific diseases are rare. A model could achieve high baseline accuracy simply by guessing "Healthy" every time. The AI agent must optimize for a Weighted F1-Score (>0.85) to ensure the model is genuinely highly sensitive to rare, crop-destroying pathologies.

Why RMSE for Growth? Root Mean Square Error provides a literal, physical measurement of our prediction error (e.g., off by 5 grams of biomass). This strict quantification is required to issue precise resource recommendations, like exact watering volumes.

Systemic Validation: The ultimate test is not whether the models predict accurately in isolation, but whether the Status Engine triggers the correct real-world advice based on those predictions.

5. Agent Directives for System Orchestration
When designing, refactoring, or extending the Demeter codebase, the AI agent must strictly adhere to the following logical rules:

Maintain Modularity: The vision component, the environmental regression component, and the status engine must remain logically decoupled. They should only meet at the orchestration layer (the inference engine). This ensures one model failing or updating does not cascade into systemic failure.

Enforce Edge-Computing Constraints: The architecture must remain lightweight. Algorithmic complexity should be pushed to the training phase, ensuring inference operations require minimal memory and compute.

Ethical Data Handling: Assume all datasets carry inherent biological biases (e.g., overrepresentation of common commercial crops). The system logic should gracefully handle uncertain edge cases rather than providing confident, incorrect diagnoses that could lead to crop loss.

Longitudinal Awareness: The system is not stateless. It must log current diagnostics to build a historical narrative of the plant's health, allowing future iterations of the model to learn from past interventions.

Signal Preprocessing Exploration (FFT): To determine if frequency or color content is more useful than purely spatial convolutions, developers should investigate performing a Discrete-Time Fourier Transform (DTFT/FFT) on images prior to classification as an experimental preprocessing step.