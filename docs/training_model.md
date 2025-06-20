Certainly! Here is a **beginner-friendly tutorial** in Markdown for the model training script (`app/core/model_training.py`) in your Travel Concierge System project.

---

# 🚀 Beginner Tutorial: Model Training for the Travel Concierge System

This guide will walk you through the basics of how the model training script works, how to use it, and how to customize it for your own travel data.

---

## Table of Contents

1. [What is this script for?](#what-is-this-script-for)
2. [How does it work?](#how-does-it-work)
3. [Key Components](#key-components)
    - [1. Initialization](#1-initialization)
    - [2. Preparing Training Data](#2-preparing-training-data)
    - [3. Formatting Examples](#3-formatting-examples)
    - [4. Fine-tuning the Model](#4-fine-tuning-the-model)
    - [5. Loading a Fine-tuned Model](#5-loading-a-fine-tuned-model)
4. [How to Train Your Model](#how-to-train-your-model)
5. [Tips & Troubleshooting](#tips--troubleshooting)
6. [Further Reading](#further-reading)

---

## What is this script for?

The script in `app/core/model_training.py` is designed to **fine-tune a language model** (like GPT-2) on travel booking data. This allows your AI assistant to better understand and respond to travel-related queries, such as booking flights, hotels, and handling user preferences.

---

## How does it work?

- **Loads a pre-trained language model** (default: GPT-2).
- **Prepares your travel data** into a format the model can learn from.
- **Fine-tunes the model** on your data using the HuggingFace Transformers library.
- **Saves the trained model** for later use in your application.

---

## Key Components

### 1. Initialization

```python
trainer = TravelModelTrainer(
    base_model="gpt2",  # You can change this to another model if you want
    output_dir="models/travel_concierge"
)
```
- Loads the base model and tokenizer.
- Adds special tokens for travel concepts (like `<flight>`, `<hotel>`, etc.).

---

### 2. Preparing Training Data

```python
dataset = trainer.prepare_training_data(training_data)
```
- Converts your list of travel examples (dictionaries) into a text format.
- Tokenizes the text so the model can understand it.
- Returns a HuggingFace `Dataset` ready for training.

---

### 3. Formatting Examples

Each training example is formatted like this:
```
<preference>
Traveler Type: Road Warrior
Preferred Airlines: Delta, United
Preferred Hotels: Marriott, Hilton
</preference>

<flight>
From: New York
To: London
Date: 2024-07-01
Airline: Delta
</flight>

<hotel>
Location: London
Check-in: 2024-07-01
Check-out: 2024-07-05
Hotel: Marriott London
</hotel>

<meeting>
Purpose: client_meeting
Time: 2024-07-02 10:00
</meeting>

<constraint>
Budget: 5000
Special Requirements: WiFi, 24h Business Center
</constraint>
```
- This structure helps the model learn to associate user preferences, flights, hotels, and constraints.

---

### 4. Fine-tuning the Model

```python
trainer.fine_tune(
    training_data=training_data,  # List of your training examples
    num_epochs=3,                 # How many times to go through the data
    batch_size=4,                 # How many samples per training step
    learning_rate=2e-5            # How fast the model learns
)
```
- Trains the model on your data.
- Saves the model and tokenizer to the output directory.

---

### 5. Loading a Fine-tuned Model

```python
model, tokenizer = trainer.load_fine_tuned_model("models/travel_concierge")
```
- Loads your trained model for inference or further training.

---

## How to Train Your Model

1. **Prepare your data**  
   Make sure you have a list of dictionaries, each representing a travel scenario.

2. **Initialize the trainer**
   ```python
   from app.core.model_training import TravelModelTrainer
   trainer = TravelModelTrainer()
   ```

3. **Train the model**
   ```python
   trainer.fine_tune(training_data)
   ```

4. **Use the trained model in your app!**

---

## Tips & Troubleshooting

- **GPU recommended:** Training is much faster with a GPU. If you get CUDA errors, make sure your environment is set up for GPU use.
- **Data format:** Ensure your training data matches the expected keys (see the `_format_example` method).
- **Model size:** GPT-2 is a good starting point. For larger datasets or more complex tasks, consider a larger model (but this requires more resources).
- **Custom tokens:** You can add more special tokens if your domain needs them.

---

## Further Reading

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Fine-tuning Language Models](https://huggingface.co/docs/transformers/training)
- [Datasets Library](https://huggingface.co/docs/datasets/index)

---

**Happy training!** If you have questions or want to extend the script, check the comments in the code or ask for help.