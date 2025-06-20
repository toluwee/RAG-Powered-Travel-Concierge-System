from typing import List, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelModelTrainer:
    def __init__(
        self,
        base_model: str = "gpt2", # gpt2 is a small model that is easy to train and use
        output_dir: str = "models/travel_concierge"
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the base model and tokenizer"""
        logger.info(f"Loading base model: {self.base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Add special tokens for travel domain
        special_tokens = {
            "additional_special_tokens": [
                "<flight>", "</flight>",
                "<hotel>", "</hotel>",
                "<meeting>", "</meeting>",
                "<preference>", "</preference>",
                "<constraint>", "</constraint>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        # Ensure pad_token is set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_training_data(self, data: List[Dict[str, Any]]) -> Dataset:
        """Prepare training data from travel booking examples"""
        # Convert data to text format
        texts = []
        for example in data:
            text = self._format_example(example)
            texts.append({"text": text})
        
        # Create dataset
        dataset = Dataset.from_pandas(pd.DataFrame(texts))
        
        # Tokenize dataset using __call__ method for fast tokenizers
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format a single training example"""
        # This is a simplified example - in practice, you'd want more sophisticated formatting
        text = f"""
        <preference>
        Traveler Type: {example.get('traveler_type', 'unknown')}
        Preferred Airlines: {', '.join(example.get('preferred_airlines', []))}
        Preferred Hotels: {', '.join(example.get('preferred_hotels', []))}
        </preference>

        <flight>
        From: {example.get('origin', 'unknown')}
        To: {example.get('destination', 'unknown')}
        Date: {example.get('date', 'unknown')}
        Airline: {example.get('airline', 'unknown')}
        </flight>

        <hotel>
        Location: {example.get('hotel_location', 'unknown')}
        Check-in: {example.get('check_in', 'unknown')}
        Check-out: {example.get('check_out', 'unknown')}
        Hotel: {example.get('hotel_name', 'unknown')}
        </hotel>

        <meeting>
        Purpose: {example.get('purpose', 'unknown')}
        Time: {example.get('meeting_time', 'unknown')}
        </meeting>

        <constraint>
        Budget: {example.get('budget', 'unknown')}
        Special Requirements: {', '.join(example.get('special_requirements', []))}
        </constraint>
        """
        return text

    def fine_tune(
        self,
        training_data: List[Dict[str, Any]],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5
    ):
        """Fine-tune the model on travel booking data"""
        logger.info("Preparing training data...")
        dataset = self.prepare_training_data(training_data)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            save_strategy="epoch",
            evaluation_strategy="no",
            load_best_model_at_end=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            # Do not pass eval_dataset if not available
        )
        
        # Start training
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model"""
        logger.info(f"Loading fine-tuned model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return self.model, self.tokenizer 