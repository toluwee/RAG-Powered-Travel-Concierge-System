# Intelligent Travel Concierge System

An AI-powered travel planning system for complex multi-city business trips, featuring fine-tuned foundation models, RAG (Retrieval-Augmented Generation), and intelligent orchestration.

## Features

- **Multi-City Trip Planning**: Handle complex business trips with multiple destinations
- **AI-Powered Recommendations**: Fine-tuned models trained on travel data
- **RAG System**: Real-time retrieval from multiple data sources
- **Context-Aware Personalization**: Adapts to traveler preferences and corporate policies
- **Intelligent Orchestration**: Agentic coordination of travel components

## Architecture

```
Travel/
├── app/                    # Main application
│   ├── api/               # FastAPI routes
│   ├── core/              # Core functionality (RAG, model training)
│   ├── models/            # Data models
│   ├── services/          # Business logic services
│   └── main.py           # FastAPI application entry point
├── data/                  # Data storage
│   ├── synthetic_training_data.json
│   ├── rag_documents.json
│   └── vector_store/     # FAISS vector store
├── models/               # Trained models
├── tests/                # Test suite
├── make.py              # Automation script
└── requirements.txt     # Dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Setup

```bash
# Generate data, setup RAG, and test the system
python make.py full
```

### 3. Start the API Server

```bash
# Start the FastAPI server
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## Usage

### Command Line Interface

The `make.py` script provides several commands:

```bash
# Generate synthetic training data and RAG documents
python make.py generate-data

# Setup RAG system with documents
python make.py setup-rag

# Train the model (requires TRAIN_MODEL=true)
TRAIN_MODEL=true python make.py train-model

# Test the complete system
python make.py test

# Run complete setup
python make.py full
```

### API Endpoints

- `POST /api/v1/travel/plan` - Plan a multi-city business trip
- `GET /api/v1/travel/{request_id}` - Get trip status and details
- `POST /api/v1/travel/{request_id}/modify` - Modify existing trip
- `GET /api/v1/recommendations` - Get personalized recommendations

### Example Request

```python
from app.models.travel import TravelRequest, Location, TripSegment, FlightSegment, HotelStay
from datetime import datetime, timedelta

# Create a multi-city business trip request
request = TravelRequest(
    traveler_type="road_warrior",
    segments=[
        TripSegment(
            flight=FlightSegment(
                origin=Location(city="Seattle", country="USA"),
                destination=Location(city="London", country="UK"),
                departure_time=datetime.now() + timedelta(days=1),
                airline="Delta"
            ),
            hotel=HotelStay(
                location=Location(city="London", country="UK"),
                check_in=datetime.now() + timedelta(days=1),
                check_out=datetime.now() + timedelta(days=3),
                hotel_name="Marriott London"
            ),
            purpose="client_meeting"
        )
    ],
    preferences={
        "preferred_airlines": ["Delta"],
        "preferred_hotel_chains": ["Marriott"],
        "requires_24h_business_center": True
    }
)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app
```

### Project Structure

- **Models** (`app/models/`): Pydantic models for data validation
- **Services** (`app/services/`): Business logic and external integrations
- **Core** (`app/core/`): RAG system and model training
- **API** (`app/api/`): FastAPI route handlers
- **Tests** (`tests/`): Test suite

## Configuration

Environment variables can be set in a `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Model Training
TRAIN_MODEL=false
MODEL_BASE_PATH=models/travel_concierge

# RAG System
RAG_VECTOR_STORE_PATH=data/vector_store
RAG_DOCUMENTS_PATH=data/rag_documents.json

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## Dependencies

- **FastAPI**: Web framework
- **LangChain**: RAG and LLM orchestration
- **Transformers**: Model fine-tuning
- **FAISS**: Vector similarity search
- **Pydantic**: Data validation
- **SQLAlchemy**: Database ORM

## License

MIT License - see LICENSE file for details. 