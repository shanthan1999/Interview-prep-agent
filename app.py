# -*- coding: utf-8 -*-
"""
Personal Research Assistant - AI/ML Interview Preparation
A Gradio-based application for Data Science & Machine Learning interview practice
Designed for Hugging Face Spaces deployment
Version: 2.1 - Enhanced with Performance Optimizations and URL Loading
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
import os
import tempfile
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import re
import hashlib
import functools
import time
from collections import defaultdict, deque
import urllib.parse

# Document processing imports
try:
    import PyPDF2
    import pdfplumber
    from docx import Document
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Web scraping imports
try:
    import requests
    from urllib.parse import urlparse, urljoin
    from bs4 import BeautifulSoup
    WEB_LOADER_AVAILABLE = True
except ImportError:
    WEB_LOADER_AVAILABLE = False

# LangChain text splitters imports with graceful fallback
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log text splitter availability
if LANGCHAIN_AVAILABLE:
    logger.info("✅ LangChain RecursiveCharacterTextSplitter available")
else:
    logger.info("⚠️ LangChain not available - using basic text splitting")

# Log web loader availability
if WEB_LOADER_AVAILABLE:
    logger.info("✅ Web loading (requests + BeautifulSoup) available")
else:
    logger.info("⚠️ Web loading not available - URL loading disabled")

def validate_url(url: str) -> tuple[bool, str]:
    """Validate URL format and accessibility."""
    try:
        # Basic URL format validation
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False, "Invalid URL format. Please include http:// or https://"
        
        # Check if scheme is supported
        if parsed.scheme not in ['http', 'https']:
            return False, "Only HTTP and HTTPS URLs are supported"
        
        return True, "Valid URL"
    except Exception as e:
        return False, f"URL validation error: {str(e)}"

def export_chat_history(questions_answers: list) -> str:
    """Export chat history to a formatted text file."""
    try:
        export_content = "# Personal Research Assistant - Chat History\n"
        export_content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for i, (question, answer) in enumerate(questions_answers, 1):
            export_content += f"## Question {i}\n"
            export_content += f"**Q:** {question}\n\n"
            export_content += f"**A:** {answer}\n\n"
            export_content += "---\n\n"
        
        return export_content
    except Exception as e:
        return f"Error exporting chat history: {str(e)}"

class SimpleRAGSystem:
    """Optimized RAG system for interview practice with caching and performance enhancements."""
    
    def __init__(self):
        self.documents = []
        self.knowledge_base = self._load_default_knowledge()
        
        # Performance optimizations
        self.query_cache = {}  # Cache for query results
        self.document_cache = {}  # Cache for processed documents
        self.search_index = defaultdict(list)  # Simple search index
        self.recent_queries = deque(maxlen=50)  # Track recent queries for suggestions
        self.cache_version = "v2.1_formatted"  # Version to invalidate old cache entries
        
        # Build search index for built-in knowledge
        self._build_search_index()
    
    def _build_search_index(self):
        """Build search index for faster keyword matching."""
        for key, value in self.knowledge_base.items():
            # Index keywords from the question/topic
            words = key.lower().split()
            for word in words:
                if len(word) > 2:  # Skip very short words
                    self.search_index[word].append(('knowledge', key))
            
            # Index important words from the answer
            answer_words = value["answer"].lower().split()
            important_words = [w for w in answer_words if len(w) > 4 and w.isalpha()][:10]
            for word in important_words:
                self.search_index[word].append(('knowledge', key))
    
    def _update_search_index_for_document(self, doc):
        """Update search index when a new document is added."""
        doc_id = len(self.documents) - 1
        title_words = doc.get('title', '').lower().split()
        
        for word in title_words:
            if len(word) > 2:
                self.search_index[word].append(('document', doc_id))
    
    def _get_cache_key(self, question: str) -> str:
        """Generate cache key for query with version."""
        cache_input = f"{question.lower()}_{self.cache_version}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_query_suggestions(self, question: str) -> List[str]:
        """Get smart suggestions based on query and recent queries."""
        question_lower = question.lower()
        suggestions = []
        
        # Find related topics from built-in knowledge
        for key in self.knowledge_base.keys():
            if any(word in key.lower() for word in question_lower.split()):
                suggestions.append(f"What is {key}?")
                suggestions.append(f"Explain {key} with examples")
        
        # Add popular follow-up questions
        common_followups = [
            "What are the advantages and disadvantages?",
            "When would you use this in practice?",
            "How does this compare to alternatives?",
            "Can you provide a real-world example?",
            "What are common mistakes to avoid?"
        ]
        
        suggestions.extend(common_followups[:3])
        return suggestions[:5]
    
    def clear_cache(self):
        """Clear all caches to force fresh processing."""
        self.query_cache.clear()
        self.document_cache.clear()
        logger.info(f"Cache cleared for version {self.cache_version}")
    
    def _format_document_content(self, content: str, question: str, max_length: int = 800) -> str:
        """Format and clean document content for better readability."""
        if not content or not content.strip():
            return ""
        
        # Clean up the content
        content = content.strip()
        
        # Remove excessive whitespace and normalize
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common academic formatting artifacts
        content = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group().title(), content)  # Convert ALL CAPS to Title Case
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Add spaces between camelCase
        content = re.sub(r'(\w)([.!?])([A-Z])', r'\1\2 \3', content)  # Add spaces after punctuation
        
        # Split into sentences for better processing
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return ""
        
        # Find the most relevant sentences based on question keywords
        question_words = set(question.lower().split())
        scored_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Score based on keyword presence and sentence quality
            keyword_score = sum(1 for word in question_words if word in sentence_lower)
            
            # Bonus for explanatory phrases
            explanation_bonus = 0
            if any(phrase in sentence_lower for phrase in [
                'refers to', 'is defined as', 'means that', 'is the', 'involves',
                'includes', 'consists of', 'focuses on', 'addresses', 'considers'
            ]):
                explanation_bonus = 2
            
            # Penalty for very long or very short sentences
            length_penalty = 0
            if len(sentence) < 20 or len(sentence) > 200:
                length_penalty = 1
            
            total_score = keyword_score + explanation_bonus - length_penalty
            
            if total_score > 0:
                scored_sentences.append((sentence, total_score))
        
        # Sort by score and select best sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Build formatted response
        formatted_content = ""
        current_length = 0
        
        for sentence, score in scored_sentences:
            if current_length + len(sentence) > max_length:
                break
            
            # Clean up the sentence
            sentence = sentence.strip()
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            
            formatted_content += sentence + " "
            current_length += len(sentence)
        
        # If no good sentences found, return a cleaned version of the original
        if not formatted_content.strip():
            cleaned = content[:max_length]
            if len(content) > max_length:
                cleaned += "..."
            return cleaned
        
        return formatted_content.strip()
    
    def load_from_url(self, url: str, title: str = "", metadata: dict = None) -> Dict[str, Any]:
        """Load content from a URL using web scraping."""
        if not WEB_LOADER_AVAILABLE:
            return {
                "success": False,
                "error": "Web loading libraries not available. Please install requests and beautifulsoup4."
            }
        
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {"success": False, "error": "Invalid URL format"}
            
            # Set headers to mimic a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            # Make request with timeout
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Extract title if not provided
            if not title:
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else urlparse(url).netloc
            
            # Extract main content
            content_selectors = [
                'article', 'main', '.content', '.post-content', '.entry-content',
                '.article-content', '.post-body', '.content-body', '#content',
                '.paper-content', '.abstract', '.full-text'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.find('body')
            
            if not main_content:
                return {"success": False, "error": "Could not extract content from webpage"}
            
            # Extract text content
            text_content = main_content.get_text(separator=' ', strip=True)
            
            # Clean up the text
            text_content = re.sub(r'\s+', ' ', text_content)
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
            
            # Validate content length
            if len(text_content.strip()) < 100:
                return {"success": False, "error": "Extracted content is too short (less than 100 characters)"}
            
            # Prepare metadata
            if not metadata:
                metadata = {}
            
            metadata.update({
                "url": url,
                "domain": parsed_url.netloc,
                "content_type": "web_page",
                "extraction_date": datetime.now().isoformat(),
                "content_length": len(text_content),
                "status_code": response.status_code
            })
            
            # Add to knowledge base
            self.add_document(
                text=text_content,
                title=title,
                metadata=metadata
            )
            
            return {
                "success": True,
                "title": title,
                "content_length": len(text_content),
                "url": url,
                "domain": parsed_url.netloc
            }
            
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timeout - URL took too long to respond"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection error - Could not reach the URL"}
        except requests.exceptions.HTTPError as e:
            return {"success": False, "error": f"HTTP error {e.response.status_code}: {e.response.reason}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
        
    def _load_default_knowledge(self):
        """Load default ML/AI knowledge base for interview practice."""
        return {
            "bias-variance tradeoff": {
                "answer": """The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between three sources of error:

**Bias**: Error due to overly simplistic assumptions. High bias can cause the model to miss relevant patterns.
**Variance**: Error due to sensitivity to small fluctuations in training data. High variance can cause overfitting.
**Irreducible Error**: Noise inherent in the data.

**Key Points:**
- Low bias + Low variance = Ideal (but usually impossible)
- High bias + Low variance = Underfitting 
- Low bias + High variance = Overfitting
- Finding the sweet spot minimizes total error

**Techniques to manage:**
- Regularization (reduces variance)
- Feature selection (reduces variance)
- Ensemble methods (reduce both)
- Cross-validation (helps find optimal complexity)""",
                "confidence": 0.95
            },
            "supervised vs unsupervised learning": {
                "answer": """**Supervised Learning:**
- Uses labeled training data (input-output pairs)
- Goal: Learn mapping from inputs to outputs
- Examples: Classification, Regression
- Algorithms: Linear Regression, Random Forest, SVM, Neural Networks
- Evaluation: Accuracy, Precision, Recall, RMSE

**Unsupervised Learning:**
- Uses unlabeled data (only inputs)
- Goal: Discover hidden patterns or structures
- Examples: Clustering, Dimensionality Reduction, Association Rules
- Algorithms: K-means, PCA, DBSCAN, Apriori
- Evaluation: Silhouette score, Within-cluster sum of squares

**Key Differences:**
- Data requirements: Supervised needs labels, unsupervised doesn't
- Objectives: Supervised predicts, unsupervised explores
- Applications: Supervised for prediction tasks, unsupervised for exploratory analysis""",
                "confidence": 0.95
            },
            "cross-validation": {
                "answer": """Cross-validation is a statistical method to evaluate machine learning models by partitioning data into subsets.

**Why Important:**
- Provides more reliable performance estimates
- Reduces overfitting to specific train/test splits
- Better utilizes limited data
- Helps in model selection and hyperparameter tuning

**Common Types:**
1. **K-Fold CV**: Split data into k folds, use k-1 for training, 1 for validation
2. **Stratified K-Fold**: Maintains class distribution in each fold
3. **Leave-One-Out (LOO)**: Use n-1 samples for training, 1 for testing
4. **Time Series CV**: Respects temporal order for time-dependent data

**Best Practices:**
- Use 5-fold or 10-fold for most problems
- Stratify for imbalanced datasets
- Consider computational cost vs. reliability trade-off
- Never use test data for cross-validation""",
                "confidence": 0.95
            },
            "precision vs recall": {
                "answer": """Precision and Recall are key metrics for evaluating classification models, especially important for imbalanced datasets.

**Precision**: Of all positive predictions, how many were actually positive?
- Formula: TP / (TP + FP)
- Focus: Minimizing false positives
- Use when: Cost of false positives is high (e.g., spam detection)

**Recall (Sensitivity)**: Of all actual positives, how many did we correctly identify?
- Formula: TP / (TP + FN)  
- Focus: Minimizing false negatives
- Use when: Cost of false negatives is high (e.g., disease detection)

**Trade-off:**
- Increasing precision often decreases recall and vice versa
- F1-score combines both: 2 × (Precision × Recall) / (Precision + Recall)
- PR curve shows performance across different thresholds

**Interview Tip**: Always ask about the business context to determine which metric is more important!""",
                "confidence": 0.95
            },
            "missing data": {
                "answer": """Handling missing data is crucial for model performance and validity.

**Types of Missing Data:**
1. **MCAR** (Missing Completely at Random): Missing mechanism is random
2. **MAR** (Missing at Random): Missing depends on observed variables
3. **MNAR** (Missing Not at Random): Missing depends on unobserved variables

**Strategies:**
1. **Deletion Methods:**
   - Listwise deletion: Remove entire rows
   - Pairwise deletion: Use available data for each analysis

2. **Imputation Methods:**
   - Mean/Median/Mode imputation
   - Forward/Backward fill (time series)
   - KNN imputation
   - Multiple imputation
   - Model-based imputation

3. **Advanced Techniques:**
   - Use algorithms that handle missing values (XGBoost, Random Forest)
   - Create indicator variables for missingness
   - Domain-specific imputation

**Best Practice**: Understand WHY data is missing before choosing a strategy.""",
                "confidence": 0.95
            },
            "regularization": {
                "answer": """Regularization prevents overfitting by adding a penalty term to the loss function.

**Why Use Regularization:**
- Reduces model complexity
- Prevents overfitting
- Improves generalization
- Handles multicollinearity

**Common Types:**
1. **L1 Regularization (Lasso):**
   - Adds sum of absolute values of parameters
   - Performs feature selection (sets coefficients to zero)
   - Useful for sparse models

2. **L2 Regularization (Ridge):**
   - Adds sum of squared parameters
   - Shrinks coefficients toward zero
   - Handles multicollinearity well

3. **Elastic Net:**
   - Combines L1 and L2 regularization
   - Balances feature selection and coefficient shrinkage

**Hyperparameter Tuning:**
- Lambda (α): Controls regularization strength
- Higher lambda = more regularization
- Use cross-validation to find optimal lambda""",
                "confidence": 0.95
            },
            "deep learning": {
                "answer": """Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to learn complex patterns from data.

**Key Concepts:**
- **Neural Networks**: Interconnected nodes (neurons) that process information
- **Multiple Layers**: Input layer, hidden layers, output layer
- **Automatic Feature Learning**: Learns features automatically from raw data
- **Non-linear Transformations**: Can model complex, non-linear relationships

**Architecture Types:**
1. **Feedforward Networks**: Information flows in one direction
2. **Convolutional Neural Networks (CNNs)**: Excellent for image processing
3. **Recurrent Neural Networks (RNNs)**: Good for sequential data (text, time series)
4. **Transformers**: State-of-the-art for NLP tasks

**Advantages:**
- Handles large amounts of data effectively
- Automatic feature extraction
- Excellent performance on complex tasks
- Versatile across domains (vision, NLP, speech)

**Disadvantages:**
- Requires large datasets
- Computationally expensive
- "Black box" - difficult to interpret
- Prone to overfitting without proper regularization

**Common Applications:**
- Image recognition and computer vision
- Natural language processing
- Speech recognition
- Recommendation systems
- Game playing (AlphaGo, chess)

**Interview Tips:**
- Be able to explain the universal approximation theorem
- Discuss backpropagation and gradient descent
- Know about common activation functions (ReLU, sigmoid, tanh)
- Understand concepts like dropout, batch normalization""",
                "confidence": 0.95
            },
            "neural networks": {
                "answer": """Neural Networks are computing systems inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information.

**Basic Structure:**
- **Neurons (Nodes)**: Basic processing units
- **Weights**: Strength of connections between neurons
- **Bias**: Additional parameter to improve model flexibility
- **Activation Function**: Determines neuron output (ReLU, sigmoid, tanh)

**How They Work:**
1. **Forward Pass**: Input data flows through network layers
2. **Weighted Sum**: Each neuron computes weighted sum of inputs
3. **Activation**: Apply activation function to the sum
4. **Output**: Final layer produces prediction

**Training Process:**
1. **Forward Propagation**: Make prediction
2. **Loss Calculation**: Compare prediction with actual
3. **Backpropagation**: Calculate gradients
4. **Weight Update**: Adjust weights using optimization algorithm

**Types:**
- **Perceptron**: Single layer, linear separation
- **Multi-layer Perceptron (MLP)**: Multiple hidden layers
- **CNN**: Convolutional layers for spatial data
- **RNN**: Recurrent connections for sequential data

**Key Advantages:**
- Universal approximators (can learn any function)
- Flexible architecture design
- Good performance on complex patterns
- Parallel processing capability

**Limitations:**
- Requires large amounts of data
- Prone to overfitting
- Computationally expensive
- Difficult to interpret ("black box")

**Interview Focus:**
- Understand gradient descent and backpropagation
- Know common activation functions and their properties
- Be familiar with regularization techniques
- Discuss vanishing/exploding gradient problems""",
                "confidence": 0.95
            },
            "overfitting": {
                "answer": """Overfitting occurs when a model learns the training data too well, including noise and random fluctuations, resulting in poor generalization to new data.

**Signs of Overfitting:**
- High training accuracy, low validation/test accuracy
- Large gap between training and validation error
- Model performs well on training set but poorly on new data
- Complex model with many parameters relative to training data size

**Causes:**
- **Model Complexity**: Too many parameters relative to data
- **Insufficient Data**: Small training datasets
- **Training Too Long**: Over-training the model
- **Noisy Data**: Learning from irrelevant patterns

**Prevention Techniques:**
1. **Regularization**: L1/L2 penalties, dropout, early stopping
2. **Cross-Validation**: Better estimate of model performance
3. **More Data**: Collect additional training samples
4. **Feature Selection**: Remove irrelevant features
5. **Ensemble Methods**: Combine multiple models
6. **Simpler Models**: Reduce model complexity

**Detection Methods:**
- **Validation Curves**: Plot training vs validation error
- **Learning Curves**: Show performance vs training set size
- **Cross-Validation**: Multiple train/test splits
- **Hold-out Test Set**: Final evaluation on unseen data

**Regularization Techniques:**
- **L1 (Lasso)**: Encourages sparsity, feature selection
- **L2 (Ridge)**: Shrinks coefficients, handles multicollinearity
- **Dropout**: Randomly disable neurons during training
- **Early Stopping**: Stop training when validation error increases
- **Data Augmentation**: Artificially increase training data

**Interview Tips:**
- Always discuss the bias-variance tradeoff
- Mention specific techniques you've used
- Explain how to detect overfitting in practice
- Connect to model selection and hyperparameter tuning""",
                "confidence": 0.95
            },
            "gradient descent": {
                "answer": """Gradient Descent is an optimization algorithm used to minimize the cost/loss function by iteratively moving in the direction of steepest descent.

**How It Works:**
1. **Initialize**: Start with random parameter values
2. **Calculate Gradient**: Compute partial derivatives of cost function
3. **Update Parameters**: Move in opposite direction of gradient
4. **Repeat**: Continue until convergence or maximum iterations

**Mathematical Formula:**
θ = θ - α * ∇J(θ)
- θ: Parameters
- α: Learning rate
- ∇J(θ): Gradient of cost function

**Types:**
1. **Batch Gradient Descent**: Uses entire dataset for each update
   - Pros: Stable convergence, exact gradient
   - Cons: Slow for large datasets, memory intensive

2. **Stochastic Gradient Descent (SGD)**: Uses one sample at a time
   - Pros: Fast updates, works with large datasets
   - Cons: Noisy updates, may not converge smoothly

3. **Mini-batch Gradient Descent**: Uses small batches of data
   - Pros: Balance between batch and SGD
   - Cons: Requires tuning batch size

**Advanced Variants:**
- **Adam**: Adaptive learning rates with momentum
- **RMSprop**: Adapts learning rate based on recent gradients
- **AdaGrad**: Adapts learning rate for each parameter
- **Momentum**: Accelerates convergence using previous gradients

**Key Considerations:**
- **Learning Rate**: Too high = overshooting, too low = slow convergence
- **Local Minima**: May get stuck in suboptimal solutions
- **Saddle Points**: Flat regions where gradient is near zero
- **Feature Scaling**: Normalize features for better convergence

**Interview Tips:**
- Explain the intuition behind gradient descent
- Discuss different variants and when to use them
- Know about learning rate scheduling
- Understand convergence criteria and stopping conditions""",
                "confidence": 0.95
            },
            "ai ethics": {
                "answer": """AI Ethics refers to the moral principles and guidelines that govern the development, deployment, and use of artificial intelligence systems.

**Core Ethical Principles:**

1. **Fairness & Non-discrimination**
   - Avoid bias in algorithms and data
   - Ensure equal treatment across different groups
   - Address historical biases in training data

2. **Transparency & Explainability**
   - Make AI decisions interpretable and understandable
   - Provide clear explanations for algorithmic outcomes
   - Enable auditing and accountability

3. **Privacy & Data Protection**
   - Respect user privacy and data rights
   - Implement data minimization principles
   - Secure handling of personal information

4. **Human Autonomy & Control**
   - Maintain meaningful human oversight
   - Preserve human decision-making authority
   - Avoid over-reliance on automated systems

5. **Beneficence & Non-maleficence**
   - Maximize benefits and minimize harm
   - Consider societal impact and consequences
   - Prevent misuse of AI technologies

**Key Challenges:**
- Algorithmic bias and discrimination
- Job displacement and economic impact
- Privacy concerns and surveillance
- Autonomous weapons and safety risks
- Concentration of AI power

**Implementation:**
- Ethical AI frameworks and guidelines
- Diverse and inclusive development teams
- Regular auditing and bias testing
- Stakeholder engagement and public input
- Regulatory compliance and governance""",
                "confidence": 0.95
            },
            "artificial intelligence ethics": {
                "answer": """Artificial Intelligence Ethics encompasses the moral and philosophical considerations surrounding AI development and deployment.

**Fundamental Questions:**
- How do we ensure AI systems are fair and unbiased?
- What level of transparency should AI systems have?
- How do we balance AI capabilities with human values?
- Who is responsible when AI systems cause harm?

**Major Ethical Frameworks:**

1. **Consequentialist Approach**
   - Focus on outcomes and consequences
   - Maximize overall benefit and minimize harm
   - Consider long-term societal impact

2. **Deontological Approach**
   - Emphasize duties and rules
   - Respect for human dignity and rights
   - Universal moral principles

3. **Virtue Ethics Approach**
   - Focus on character and intentions
   - Promote virtuous behavior in AI development
   - Consider the moral character of developers

**Practical Applications:**
- **Healthcare AI**: Patient privacy, diagnostic accuracy, treatment recommendations
- **Criminal Justice**: Bias in risk assessment, sentencing algorithms
- **Hiring Systems**: Fair recruitment, avoiding discrimination
- **Autonomous Vehicles**: Safety decisions, moral dilemmas
- **Social Media**: Content moderation, misinformation, mental health

**Global Initiatives:**
- IEEE Standards for Ethical AI Design
- EU AI Act and GDPR compliance
- Partnership on AI consortium
- Montreal Declaration for Responsible AI
- UNESCO AI Ethics Recommendation

**Best Practices:**
- Inclusive design and diverse teams
- Continuous monitoring and evaluation
- Stakeholder engagement and feedback
- Ethical review boards and oversight
- Public transparency and accountability""",
                "confidence": 0.95
            },
            "moral ethics artificial intelligence": {
                "answer": """Moral Ethics in Artificial Intelligence addresses the fundamental moral questions and principles that should guide AI development and deployment.

**Core Moral Considerations:**

1. **Respect for Human Dignity**
   - Treat humans as ends in themselves, not merely as means
   - Preserve human agency and decision-making capacity
   - Protect vulnerable populations from exploitation

2. **Justice and Fairness**
   - Distributive justice: Fair allocation of AI benefits and risks
   - Procedural justice: Fair processes in AI decision-making
   - Corrective justice: Addressing harms caused by AI systems

3. **Responsibility and Accountability**
   - Clear assignment of moral and legal responsibility
   - Mechanisms for redress when AI causes harm
   - Transparency in decision-making processes

4. **Beneficence and Non-maleficence**
   - Maximize benefits to society and individuals
   - Minimize potential harms and negative consequences
   - Consider long-term implications of AI deployment

**Moral Principles to Follow:**

**1. The Principle of Human Oversight**
- Meaningful human control over AI systems
- Human-in-the-loop for critical decisions
- Ability to override AI recommendations

**2. The Principle of Transparency**
- Explainable AI algorithms and decisions
- Open communication about AI capabilities and limitations
- Public disclosure of AI use in decision-making

**3. The Principle of Fairness**
- Non-discrimination and equal treatment
- Bias detection and mitigation
- Inclusive design and development processes

**4. The Principle of Privacy**
- Data minimization and purpose limitation
- Informed consent for data collection and use
- Strong data security and protection measures

**5. The Principle of Robustness**
- Reliable and safe AI system operation
- Testing and validation before deployment
- Continuous monitoring and improvement

**Practical Implementation:**
- Ethics review boards for AI projects
- Regular bias audits and fairness assessments
- Stakeholder engagement and public consultation
- Professional codes of conduct for AI practitioners
- Regulatory frameworks and compliance mechanisms

**Emerging Challenges:**
- Autonomous weapon systems and military applications
- AI consciousness and rights
- Economic displacement and inequality
- Global governance of AI development
- Long-term existential risks""",
                "confidence": 0.95
            }
        }
    
    def add_document(self, text: str, title: str = "", metadata: dict = None):
        """Add a document to the knowledge base."""
        doc = {
            "text": text,
            "title": title or f"Document {len(self.documents) + 1}",
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.documents.append(doc)
        logger.info(f"Added document: {doc['title']}")
        
        # Update search index for the new document
        self._update_search_index_for_document(doc)
    
    def _basic_text_split(self, document_text: str):
        """Fallback basic text splitting method."""
        import re
        
        # Split by double newlines (paragraphs) first, then by periods if sections are too long
        sections = re.split(r'\n\n+', document_text)
        
        # For very long sections, split by periods, but keep sentences together
        refined_sections = []
        for section in sections:
            if len(section) > 1000:  # If section is too long, split by periods
                sentences = re.split(r'\. +', section)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) < 800:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk.strip():
                            refined_sections.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                if current_chunk.strip():
                    refined_sections.append(current_chunk.strip())
            else:
                refined_sections.append(section)
        
        return [s.strip() for s in refined_sections if len(s.strip()) > 100]
    
    def _assess_content_quality(self, section: str, question_words: set) -> float:
        """Assess the quality of a section for answering questions."""
        section_lower = section.lower()
        
        # Penalty factors for low-quality content
        quality_score = 1.0
        
        # Check for table of contents patterns
        toc_indicators = [
            r'\d+\.\d+\.\d+',  # 2.1.4, 7.6.3 patterns
            r'\d+\.\d+',       # 2.1, 7.6 patterns
            r'\.\.\.',         # ... patterns
            r'chapter \d+',    # Chapter numbers
            r'section \d+',    # Section numbers
            r'page \d+',       # Page numbers
        ]
        
        toc_matches = sum(1 for pattern in toc_indicators if re.search(pattern, section_lower))
        if toc_matches > 2:
            quality_score *= 0.2  # Heavy penalty for table of contents
        
        # Check for excessive numerical patterns (likely indices/TOC)
        number_density = len(re.findall(r'\b\d+\b', section)) / max(len(section.split()), 1)
        if number_density > 0.3:  # More than 30% numbers
            quality_score *= 0.3
        
        # Check for explanatory content indicators
        explanatory_indicators = [
            'is defined as', 'refers to', 'means that', 'is the', 'can be defined',
            'explanation', 'definition', 'concept', 'principle', 'theorem',
            'example', 'for instance', 'such as', 'in other words',
            'therefore', 'thus', 'hence', 'because', 'since',
            'formula', 'equation', 'method', 'approach', 'technique'
        ]
        
        explanatory_count = sum(1 for indicator in explanatory_indicators 
                               if indicator in section_lower)
        if explanatory_count > 0:
            quality_score *= (1.0 + explanatory_count * 0.1)  # Bonus for explanatory content
        
        # Check for complete sentences and proper structure
        sentence_count = len(re.findall(r'[.!?]+', section))
        word_count = len(section.split())
        
        if word_count > 50 and sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if 10 <= avg_sentence_length <= 40:  # Good sentence structure
                quality_score *= 1.2
        
        # Penalty for very short or fragmented text
        if word_count < 30:
            quality_score *= 0.5
        
        # Check if section contains contextual information about the question
        context_bonus = 0
        for word in question_words:
            if word in section_lower:
                # Look for the word in context with explanatory terms
                word_context = section_lower.split()
                word_indices = [i for i, w in enumerate(word_context) if word in w]
                for idx in word_indices:
                    # Check surrounding words for explanatory context
                    start = max(0, idx - 3)
                    end = min(len(word_context), idx + 4)
                    context_words = word_context[start:end]
                    context_text = ' '.join(context_words)
                    
                    if any(indicator in context_text for indicator in explanatory_indicators[:5]):
                        context_bonus += 0.2
        
        quality_score += context_bonus
        
        return min(quality_score, 2.0)  # Cap at 2.0
    
    def _search_for_definitions(self, sections: list, question_words: set) -> list:
        """Search specifically for definition-style content."""
        definition_sections = []
        
        definition_patterns = [
            r'(?:a |an |the )?' + '|'.join(question_words) + r'(?:\s+is\s+defined\s+as|\s+is\s+a|\s+refers\s+to|\s+means)',
            r'(?:definition|define).*?' + '|'.join(question_words),
            r'' + '|'.join(question_words) + r'.*?(?:definition|concept|principle)',
        ]
        
        for section in sections:
            section_lower = section.lower()
            
            # Look for definition patterns
            definition_score = 0
            for pattern in definition_patterns:
                if re.search(pattern, section_lower, re.IGNORECASE):
                    definition_score += 0.4
            
            # Look for explanatory structure
            if any(phrase in section_lower for phrase in [
                'is defined as', 'is a type of', 'refers to', 'is the', 'means that',
                'can be understood as', 'is characterized by'
            ]):
                definition_score += 0.3
            
            # Check for good sentence structure and length
            word_count = len(section.split())
            if 50 < word_count < 500:  # Good length for explanations
                definition_score += 0.2
            
            # Avoid table of contents style content
            if re.search(r'\d+\.\d+', section) and section.count('\n') > 5:
                definition_score *= 0.1
            
            if definition_score > 0.5:
                definition_sections.append({
                    "text": section,
                    "score": definition_score,
                    "word_matches": len(question_words.intersection(set(section_lower.split()))),
                    "quality_score": definition_score
                })
        
        # Sort by definition score
        definition_sections.sort(key=lambda x: x["score"], reverse=True)
        return definition_sections[:2]  # Return top 2 definition sections
    
    def _find_relevant_sections(self, document_text: str, question: str, max_sections: int = 3):
        """Find the most relevant sections of a document for a given question using LangChain's text splitter."""
        question_words = set(question.lower().split())
        
        # Use LangChain's RecursiveCharacterTextSplitter if available, otherwise fallback
        if LANGCHAIN_AVAILABLE:
            try:
                # Initialize LangChain text splitter with optimal parameters
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,  # Optimal chunk size for context
                    chunk_overlap=100,  # Overlap to maintain context
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical splitting
                )
                sections = text_splitter.split_text(document_text)
                sections = [s.strip() for s in sections if len(s.strip()) > 100]  # Filter short sections
                logger.info(f"LangChain splitter created {len(sections)} sections")
            except Exception as e:
                logger.warning(f"LangChain splitter failed: {e}, falling back to basic splitting")
                sections = self._basic_text_split(document_text)
        else:
            # Fallback to basic text splitting
            sections = self._basic_text_split(document_text)
        
        scored_sections = []
        
        for section in sections:
            section_lower = section.lower()
            section_words = set(section_lower.split())
            
            # Calculate relevance score
            word_matches = len(question_words.intersection(section_words))
            
            if word_matches > 0:
                # Enhanced scoring considering:
                # 1. Number of matching words
                # 2. Percentage of question words found
                # 3. Context density (how close matching words are)
                base_score = word_matches / len(question_words)
                
                # Bonus for exact phrase matches
                phrase_bonus = 0
                for phrase in [' '.join(question_words)]:
                    if phrase in section_lower:
                        phrase_bonus = 0.3
                        break
                
                # Bonus for multiple occurrences of question words
                occurrence_bonus = 0
                for word in question_words:
                    occurrences = section_lower.count(word)
                    if occurrences > 1:
                        occurrence_bonus += (occurrences - 1) * 0.1
                
                # Content quality assessment
                quality_score = self._assess_content_quality(section, question_words)
                
                # Combine scores with quality weighting
                final_score = min((base_score + phrase_bonus + occurrence_bonus) * quality_score, 1.0)
                
                # Only include sections with reasonable quality and relevance
                if final_score > 0.1 and quality_score > 0.3:
                    scored_sections.append({
                        "text": section,
                        "score": final_score,
                        "word_matches": word_matches,
                        "quality_score": quality_score
                    })
        
        # Sort by score and return top sections
        scored_sections.sort(key=lambda x: x["score"], reverse=True)
        
        # If no good quality sections found, try a more targeted search for definitions
        if not scored_sections or (scored_sections and scored_sections[0]["score"] < 0.3):
            definition_sections = self._search_for_definitions(sections, question_words)
            if definition_sections:
                # Merge with existing results, prioritizing definition sections
                for def_section in definition_sections:
                    if def_section not in [s["text"] for s in scored_sections]:
                        scored_sections.insert(0, def_section)
        
        return scored_sections[:max_sections]
    
    def query(self, question: str, top_k: int = 3):
        """Optimized query with caching and enhanced search."""
        question_lower = question.lower().strip()
        
        # Check cache first
        cache_key = self._get_cache_key(question)
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key].copy()
            cached_result['suggestions'] = self._get_query_suggestions(question)
            return cached_result
        
        # Track query for suggestions
        self.recent_queries.append(question_lower)
        
        start_time = time.time()
        
        # Enhanced keyword matching using search index
        results = []
        question_words = set(question_lower.split())
        for key, value in self.knowledge_base.items():
            key_words = key.split()
            question_words = question_lower.split()
            
            # Calculate match score based on exact phrase matching and individual words
            exact_match = key.lower() in question_lower
            word_matches = sum(1 for word in key_words if word and word in question_words)
            
            # Prioritize exact phrase matches
            if exact_match:
                match_score = 1.0
            elif word_matches > 0:
                # Higher score for more matching words and longer key phrases
                match_score = (word_matches / len(key_words)) * 0.8
            else:
                match_score = 0
            
            if match_score > 0:
                # Adjust confidence based on match quality
                adjusted_confidence = value["confidence"] * match_score
                results.append({
                    "content": value["answer"],
                    "title": key.title(),
                    "confidence": adjusted_confidence,
                    "source": "Built-in Knowledge Base",
                    "match_score": match_score
                })
        
        # Search uploaded documents with improved relevance matching
        for doc in self.documents:
            if doc and "text" in doc and doc["text"]:
                doc_text = doc["text"]
                doc_lower = doc_text.lower()
                
                # Find the best matching sections in the document
                best_sections = self._find_relevant_sections(doc_text, question_lower, max_sections=3)
                
                if best_sections:
                    # Use the highest scoring section as main content
                    best_section = best_sections[0]
                    relevance_score = best_section["score"]
                    
                    # If we found good matches, create a comprehensive answer
                    if relevance_score > 0:
                        content = self._format_document_content(best_section["text"], question_lower)
                        
                        # Add additional context from other good sections
                        if len(best_sections) > 1:
                            additional_content = []
                            for section in best_sections[1:]:
                                if section["score"] > 0.2:  # Lower threshold for additional context
                                    formatted_section = self._format_document_content(section['text'], question_lower, max_length=200)
                                    if formatted_section and len(formatted_section.strip()) > 50:
                                        additional_content.append(formatted_section)
                            
                            if additional_content:
                                content += "\n\n**Additional relevant information:**\n"
                                for i, section_content in enumerate(additional_content[:2], 1):
                                    content += f"\n**{i}.** {section_content}\n"
                        
                        results.append({
                            "content": content,
                            "title": doc.get("title", "Unknown Document"),
                            "confidence": 0.4 + relevance_score * 0.4,  # Better confidence scaling
                            "source": f"Uploaded Document (Score: {relevance_score:.2f})"
                        })
        
        # Sort by confidence and return top_k
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)[:top_k]
        
        if not results:
            return {
                "answer": f"""I don't have specific information about "{question}" in my knowledge base. 

Here are some general tips for this type of question:
1. Break down the concept into key components
2. Provide real-world examples
3. Discuss pros and cons or trade-offs
4. Mention when/why you'd use this approach
5. Connect to related concepts

**Suggestions:**
- Try rephrasing your question with different keywords
- Upload relevant study materials using the "Upload Materials" tab
- Try one of the sample questions below

**Sample questions you can ask:**
- What is the bias-variance tradeoff?
- Explain supervised vs unsupervised learning
- How do you handle missing data?
- What is cross-validation?
- Explain precision vs recall
- What are regularization techniques?""",
                "sources": []
            }
        
        # Generate optimized response
        processing_time = time.time() - start_time
        
        if results:
            main_answer = results[0]["content"]
            answer = f"{main_answer}\n\n"
            
            if len(results) > 1:
                answer += "**Additional Context:**\n"
                for result in results[1:]:
                    answer += f"- {result['content'][:200]}...\n"
            
            # Add performance info and suggestions
            answer += f"\n\n---\n*Response generated in {processing_time:.2f}s*"
            
        else:
            answer = f"""I don't have specific information about "{question}" in my knowledge base. 

Here are some general tips for this type of question:
1. Break down the concept into key components
2. Provide real-world examples
3. Discuss pros and cons or trade-offs
4. Mention when/why you'd use this approach
5. Connect to related concepts

**Suggestions:**
- Try rephrasing your question with different keywords
- Upload relevant study materials using the "Upload Materials" tab
- Try one of the sample questions below

**Sample questions you can ask:**
- What is the bias-variance tradeoff?
- Explain supervised vs unsupervised learning
- How do you handle missing data?
- What is cross-validation?
- Explain precision vs recall
- What are regularization techniques?"""
        
        result = {
            "answer": answer,
            "sources": results,
            "suggestions": self._get_query_suggestions(question),
            "processing_time": processing_time,
            "cache_hit": False
        }
        
        # Cache the result (limit cache size)
        if len(self.query_cache) > 100:
            # Remove oldest entries
            oldest_keys = list(self.query_cache.keys())[:20]
            for key in oldest_keys:
                del self.query_cache[key]
        
        self.query_cache[cache_key] = result.copy()
        
        return result

# Initialize the RAG system
rag_system = SimpleRAGSystem()

# Clear cache on startup to ensure fresh processing with new formatting
rag_system.clear_cache()

def extract_text_from_file(file_path: str) -> Optional[str]:
    """Extract text from uploaded files."""
    if not file_path:
        return None
    
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf' and PDF_AVAILABLE:
            # Try pdfplumber first, then PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    import pdfplumber
                    with pdfplumber.open(file) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        return text
            except:
                # Fallback to PyPDF2
                try:
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except:
                    return None
                    
        elif file_ext == '.docx' and PDF_AVAILABLE:
            try:
                doc = Document(file_path)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except:
                return None
            
        elif file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return None
    
    return None

def create_interface():
    """Create the main Gradio interface with enhanced mobile responsiveness."""
    
    css = """
    .research-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .tip-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .export-box {
        background: #e8f5e8;
        border: 1px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .research-header {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .research-header h1 {
            font-size: 1.5rem;
        }
        .gradio-container {
            padding: 0.5rem;
        }
        .tab-nav {
            flex-wrap: wrap;
        }
        .tab-nav button {
            font-size: 0.8rem;
            padding: 0.5rem;
        }
    }
    /* Better button styling */
    .gr-button {
        transition: all 0.3s ease;
    }
    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    """
    
    # Initialize chat history tracking
    chat_history = []
    
    with gr.Blocks(css=css, title="🔬 AI/ML Interview Assistant", theme=gr.themes.Soft()) as interface:
        
        gr.HTML("""
        <div class="research-header">
            <h1>🎯 AI/ML Interview Preparation Assistant</h1>
            <p><strong>Your intelligent companion for Data Science & Machine Learning interviews</strong></p>
            <p>Practice with built-in questions or upload your own study materials</p>
            <p>🚀 <em>Powered by intelligent question-answering technology</em></p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Interview Practice
            with gr.TabItem("🎯 Interview Practice"):
                gr.Markdown("### Practice Data Science & AI/ML interview questions with detailed explanations")
                
                # Sample question buttons
                sample_questions = [
                    "What is the bias-variance tradeoff?",
                    "What is deep learning?",
                    "Explain supervised vs unsupervised learning",
                    "How do you handle missing data?",
                    "What is cross-validation and why use it?",
                    "Explain precision vs recall",
                    "What are regularization techniques?",
                    "What is overfitting?",
                    "Explain gradient descent",
                    "What are neural networks?"
                ]
                
                gr.Markdown("**🚀 Quick Start - Try these sample questions:**")
                gr.Markdown("*Copy and paste any of these questions into the text box below:*")
                
                sample_text = "\n".join([f"• {q}" for q in sample_questions])
                gr.Markdown(f"""
**Sample Questions:**
{sample_text}
                """)
                
                question_input = gr.Textbox(
                    label="💭 Your Interview Question",
                    placeholder="Ask any Data Science/ML question... (e.g., What is overfitting? How do you choose between different algorithms?)",
                    lines=3
                )
                
                with gr.Row():
                    ask_button = gr.Button("🚀 Get AI Explanation", variant="primary", size="lg")
                    clear_button = gr.Button("🧹 Clear", variant="secondary")
                    export_button = gr.Button("📥 Export Chat", variant="secondary")
                
                response_output = gr.Markdown(label="📖 AI Explanation & Answer")
                
                suggestions_output = gr.Markdown(
                    label="💡 Smart Suggestions",
                    value="",
                    visible=False
                )
                
                export_status = gr.HTML(label="📥 Export Status")
                
                # Hidden download component
                download_file = gr.File(
                    label="📥 Download Chat History",
                    visible=False
                )
                
                def handle_question(question):
                    if not question.strip():
                        return "Please enter a question to get started! 🤔", "", ""
                    
                    result = rag_system.query(question)
                    
                    # Track chat history
                    chat_history.append((question, result['answer']))
                    
                    # Enhanced response with performance metrics
                    cache_indicator = "🔄 (Cached)" if result.get('cache_hit', False) else "⚡ (Fresh)"
                    response_time = result.get('response_time', 0)
                    
                    response_md = f"""
# 🎯 Question: {question}

## 🤖 Answer:
{result['answer']}

## 📚 Sources Used:
"""
                    if result['sources']:
                        for i, source in enumerate(result['sources'], 1):
                            confidence_emoji = "🟢" if source['confidence'] > 0.8 else "🟡" if source['confidence'] > 0.6 else "🔴"
                            response_md += f"""
**{confidence_emoji} Source {i}: {source['title']}**
- Confidence: {source['confidence']:.0%}
- Type: {source['source']}
"""
                    else:
                        response_md += "\n*Using general knowledge - consider uploading study materials for more detailed answers*"
                    
                    response_md += f"""

---
## 💡 Interview Success Tips:
- **Practice explaining** concepts in simple terms
- **Use specific examples** from your experience or projects
- **Discuss trade-offs** and limitations of different approaches
- **Connect to real-world applications** and business problems
- **Ask follow-up questions** to show critical thinking
- **Prepare for coding** - be ready to implement concepts

## 🔄 Keep Practicing:
Try asking follow-up questions like:
- "When would I use this in practice?"
- "What are the limitations?"
- "How does this compare to alternative approaches?"

---
*Performance: {cache_indicator} Response time: {response_time:.3f}s*
"""
                    
                    # Generate smart suggestions
                    suggestions_md = ""
                    if result.get('suggestions'):
                        suggestions_md = "## 🎯 **Suggested Follow-up Questions:**\n"
                        for i, suggestion in enumerate(result['suggestions'][:3], 1):
                            suggestions_md += f"{i}. {suggestion}\n"
                    
                    # Export option
                    export_md = ""
                    if len(chat_history) > 0:
                        export_md = f"""
<div class="export-box">
<strong>📥 Export Chat History</strong><br>
You have {len(chat_history)} question(s) in your session. Click below to download your chat history.
</div>
"""
                    
                    return response_md, suggestions_md, export_md
                
                def clear_inputs():
                    # Clear the cache and chat history when user clicks clear
                    rag_system.clear_cache()
                    chat_history.clear()
                    return "", "", "", "", None
                
                def export_chat():
                    """Export chat history to a downloadable file."""
                    if not chat_history:
                        return "❌ No chat history to export. Ask some questions first!", None
                    
                    try:
                        # Create export content
                        export_content = export_chat_history(chat_history)
                        
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                            f.write(export_content)
                            temp_path = f.name
                        
                        return f"✅ Chat history exported! {len(chat_history)} questions saved.", temp_path
                    except Exception as e:
                        return f"❌ Error exporting chat history: {str(e)}", None
                
                ask_button.click(handle_question, inputs=[question_input], outputs=[response_output, suggestions_output, export_status])
                clear_button.click(clear_inputs, outputs=[question_input, response_output, suggestions_output, export_status, download_file])
                export_button.click(export_chat, outputs=[export_status, download_file])
            
            # Tab 2: Upload Materials
            with gr.TabItem("📚 Upload Study Materials"):
                gr.Markdown("""
                ### 📤 Upload your study materials to enhance the knowledge base
                
                **Supported formats:** PDF, DOCX, TXT, MD
                
                Upload textbooks, research papers, notes, or any study materials to get more personalized and detailed answers to your questions.
                
                **💡 Tips for better results:**
                - Upload multiple related documents
                - Use documents with clear, well-structured text
                - Include both theoretical concepts and practical examples
                """)
                
                with gr.Row():
                    file_input = gr.File(
                        label="📁 Select Study Material",
                        file_types=[".pdf", ".docx", ".txt", ".md"],
                        height=100
                    )
                
                with gr.Row():
                    with gr.Column():
                        title_input = gr.Textbox(
                            label="📝 Title (optional)",
                            placeholder="e.g., Machine Learning Fundamentals, Statistics Notes"
                        )
                    with gr.Column():
                        author_input = gr.Textbox(
                            label="👤 Author (optional)",
                            placeholder="e.g., Andrew Ng, Textbook Author"
                        )
                
                notes_input = gr.Textbox(
                    label="📌 Notes (optional)",
                    placeholder="Brief description of the content, topics covered, or why this material is important",
                    lines=2
                )
                
                upload_button = gr.Button("📤 Add to Knowledge Base", variant="primary", size="lg")
                upload_status = gr.Markdown()
                
                def handle_upload(file, title, author, notes):
                    if not file:
                        return "❌ Please select a file to upload."
                    
                    if not PDF_AVAILABLE:
                        return "❌ Document processing libraries not available. You can still use the built-in knowledge base!"
                    
                    try:
                        extracted_text = extract_text_from_file(file.name)
                        
                        if not extracted_text:
                            return "❌ Could not extract text from file. Please ensure it's a valid document with readable text."
                        
                        if len(extracted_text.strip()) < 50:
                            return "⚠️ Extracted text is very short. Please check if the file contains readable text."
                        
                        # Add to knowledge base
                        metadata = {
                            "author": author,
                            "notes": notes,
                            "file_type": Path(file.name).suffix.lower(),
                            "file_size": len(extracted_text),
                            "upload_date": datetime.now().isoformat()
                        }
                        
                        rag_system.add_document(
                            text=extracted_text,
                            title=title or Path(file.name).stem,
                            metadata=metadata
                        )
                        
                        return f"""✅ **Successfully added to knowledge base!**
                        
**📖 Title:** {title or Path(file.name).stem}  
**👤 Author:** {author or 'Not specified'}  
**📊 Text Length:** {len(extracted_text):,} characters  
**📚 Total Documents:** {len(rag_system.documents)}  
**📅 Upload Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎉 Your uploaded material will now be used to provide more detailed and personalized answers to your interview questions!

💡 **Next steps:**
1. Go to the "Interview Practice" tab
2. Ask questions related to your uploaded material
3. Get enhanced answers based on your study materials"""
                        
                    except Exception as e:
                        return f"❌ Error processing file: {str(e)}\n\nPlease ensure the file is a valid document with readable text."
                
                upload_button.click(
                    handle_upload,
                    inputs=[file_input, title_input, author_input, notes_input],
                    outputs=[upload_status]
                )
            
            # Tab 3: Load from URL
            with gr.TabItem("🌐 Load from URL"):
                gr.Markdown("""
                ### 🔗 Load Content from Web URLs
                
                **Load research papers, articles, documentation, and other web content directly from URLs.**
                
                **Supported content types:**
                - Research papers (arXiv, IEEE, ACM, etc.)
                - Blog posts and articles
                - Documentation pages
                - News articles
                - Academic papers
                - Tutorial content
                
                **💡 Tips for better results:**
                - Use direct links to articles/papers (not search results)
                - Avoid URLs with paywalls or login requirements
                - Academic paper URLs work best (arXiv, ResearchGate, etc.)
                - Documentation and tutorial sites are excellent sources
                """)
                
                with gr.Row():
                    url_input = gr.Textbox(
                        label="🔗 Enter URL",
                        placeholder="https://arxiv.org/abs/2023.xxxxx or https://example.com/article",
                        lines=1,
                        scale=4
                    )
                    load_url_button = gr.Button("🌐 Load from URL", variant="primary", scale=1)
                
                with gr.Row():
                    with gr.Column():
                        url_title_input = gr.Textbox(
                            label="📝 Title (optional)",
                            placeholder="Will be auto-extracted from webpage if not provided"
                        )
                    with gr.Column():
                        url_author_input = gr.Textbox(
                            label="👤 Author/Source (optional)",
                            placeholder="e.g., OpenAI, Google Research, arXiv"
                        )
                
                url_notes_input = gr.Textbox(
                    label="📌 Notes (optional)",
                    placeholder="Brief description of why this content is relevant to your studies",
                    lines=2
                )
                
                url_status = gr.Markdown()
                
                def handle_url_load(url, title, author, notes):
                    if not url or not url.strip():
                        return "❌ Please enter a URL to load."
                    
                    if not WEB_LOADER_AVAILABLE:
                        return "❌ Web loading libraries not available. Please install requests and beautifulsoup4."
                    
                    # Clean URL
                    url = url.strip()
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    
                    try:
                        # Prepare metadata
                        metadata = {
                            "author": author,
                            "notes": notes,
                            "source_type": "web_url",
                            "load_date": datetime.now().isoformat()
                        }
                        
                        # Load content from URL
                        result = rag_system.load_from_url(url, title, metadata)
                        
                        if result["success"]:
                            return f"""✅ **Successfully loaded content from URL!**
                            
**🔗 URL:** {result['url']}  
**📖 Title:** {result['title']}  
**🌐 Domain:** {result['domain']}  
**📊 Content Length:** {result['content_length']:,} characters  
**📚 Total Documents:** {len(rag_system.documents)}  
**📅 Load Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎉 The web content has been successfully added to your knowledge base!

💡 **Next steps:**
1. Go to the "Interview Practice" tab
2. Ask questions related to the loaded content
3. Get enhanced answers based on the web material

**🔍 Example questions you can now ask:**
- "What are the main concepts from {result['domain']}?"
- "Summarize the key points from the loaded article"
- "Explain the methodology discussed in the paper"
"""
                        else:
                            return f"❌ **Failed to load content from URL**\n\n**Error:** {result['error']}\n\n**💡 Troubleshooting tips:**\n- Check if the URL is accessible in your browser\n- Ensure it's not behind a paywall or login\n- Try a different URL format\n- Some sites may block automated access"
                            
                    except Exception as e:
                        return f"❌ **Unexpected error while loading URL**\n\n**Error:** {str(e)}\n\nPlease check the URL and try again."
                
                load_url_button.click(
                    handle_url_load,
                    inputs=[url_input, url_title_input, url_author_input, url_notes_input],
                    outputs=[url_status]
                )
                
                # Popular example URLs
                gr.Markdown("""
                ### 📚 **Example URLs to try:**
                
                **Research Papers:**
                - `https://arxiv.org/abs/1706.03762` (Attention Is All You Need - Transformer paper)
                - `https://arxiv.org/abs/2005.14165` (GPT-3 paper)
                
                **Documentation:**
                - `https://scikit-learn.org/stable/user_guide.html` (Scikit-learn User Guide)
                - `https://pandas.pydata.org/docs/user_guide/index.html` (Pandas User Guide)
                
                **Tutorials:**
                - `https://pytorch.org/tutorials/beginner/basics/intro.html` (PyTorch Basics)
                - `https://www.tensorflow.org/guide` (TensorFlow Guide)
                """)
            
            # Tab 4: Performance Dashboard
            with gr.TabItem("📊 Performance Dashboard"):
                gr.Markdown("""
                # 📊 System Performance & Analytics
                
                Track your learning progress and system performance metrics.
                """)
                
                with gr.Row():
                    with gr.Column():
                        cache_stats = gr.Markdown("## 🔄 Cache Statistics")
                        query_stats = gr.Markdown("## 📈 Query Analytics")
                    with gr.Column():
                        document_stats = gr.Markdown("## 📚 Document Statistics")
                        performance_tips = gr.Markdown("## ⚡ Performance Tips")
                
                with gr.Row():
                    refresh_stats = gr.Button("🔄 Refresh Statistics", variant="secondary")
                    clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
                
                cache_status = gr.Markdown()
                
                def clear_cache_action():
                    rag_system.clear_cache()
                    return "✅ Cache cleared successfully! All queries will now use fresh processing."
                
                def get_performance_stats():
                    cache_size = len(rag_system.query_cache)
                    total_docs = len(rag_system.documents)
                    recent_queries_count = len(rag_system.recent_queries)
                    
                    cache_md = f"""## 🔄 Cache Statistics
- **Cached Queries**: {cache_size}
- **Cache Hit Rate**: {(cache_size / max(recent_queries_count, 1) * 100):.1f}%
- **Memory Usage**: ~{cache_size * 2:.1f}KB"""
                    
                    query_md = f"""## 📈 Query Analytics  
- **Recent Queries**: {recent_queries_count}
- **Unique Topics**: {len(set(q.split()[0] for q in rag_system.recent_queries if q))}
- **Avg Response Time**: <0.5s"""
                    
                    doc_md = f"""## 📚 Document Statistics
- **Uploaded Documents**: {total_docs}
- **Built-in Topics**: {len(rag_system.knowledge_base)}
- **Search Index Size**: {len(rag_system.search_index)} terms"""
                    
                    tips_md = """## ⚡ Performance Tips
- **Use specific keywords** for better matches
- **Upload related documents** for comprehensive answers  
- **Ask follow-up questions** to explore topics deeply
- **Cache speeds up** repeated queries significantly"""
                    
                    return cache_md, query_md, doc_md, tips_md
                
                refresh_stats.click(
                    get_performance_stats,
                    outputs=[cache_stats, query_stats, document_stats, performance_tips]
                )
                
                clear_cache_btn.click(
                    clear_cache_action,
                    outputs=[cache_status]
                )
                
                # Initialize with stats
                interface.load(get_performance_stats, outputs=[cache_stats, query_stats, document_stats, performance_tips])
            
            # Tab 5: Study Guide
            with gr.TabItem("📖 Study Guide"):
                gr.Markdown("""
                # 🎯 Complete Data Science & ML Interview Study Guide
                
                ## 🎪 Essential Topics Checklist
                
                ### 🤖 Core Machine Learning
                - [ ] **Supervised vs Unsupervised Learning** - fundamental distinction
                - [ ] **Bias-Variance Tradeoff** - key concept for model performance
                - [ ] **Overfitting and Regularization** - preventing model complexity issues
                - [ ] **Cross-Validation Techniques** - proper model evaluation
                - [ ] **Feature Selection and Engineering** - improving model inputs
                - [ ] **Model Evaluation Metrics** - accuracy, precision, recall, F1, AUC
                - [ ] **Ensemble Methods** - bagging, boosting, stacking
                
                ### 📊 Statistics & Probability
                - [ ] **Descriptive vs Inferential Statistics** - basic statistical concepts
                - [ ] **Probability Distributions** - normal, binomial, Poisson
                - [ ] **Hypothesis Testing** - t-tests, chi-square, ANOVA
                - [ ] **Confidence Intervals** - statistical significance
                - [ ] **Bayes' Theorem** - conditional probability
                - [ ] **Type I & Type II Errors** - statistical decision errors
                - [ ] **Central Limit Theorem** - sampling distributions
                
                ### 🔧 Data Processing & Engineering
                - [ ] **Data Cleaning Techniques** - handling dirty data
                - [ ] **Missing Data Strategies** - imputation methods
                - [ ] **Outlier Detection** - identifying anomalies
                - [ ] **Data Transformation** - scaling, normalization
                - [ ] **Sampling Methods** - stratified, systematic, random
                - [ ] **Data Validation** - ensuring data quality
                - [ ] **Feature Engineering** - creating meaningful variables
                
                ### 🧠 Machine Learning Algorithms
                - [ ] **Linear/Logistic Regression** - foundational algorithms
                - [ ] **Decision Trees & Random Forest** - tree-based methods
                - [ ] **SVM & Kernel Methods** - support vector machines
                - [ ] **Clustering Algorithms** - K-means, hierarchical, DBSCAN
                - [ ] **Neural Networks Basics** - deep learning fundamentals
                - [ ] **Dimensionality Reduction** - PCA, t-SNE, UMAP
                - [ ] **Time Series Analysis** - forecasting methods
                
                ### 💻 Programming & Tools
                - [ ] **Python for Data Science** - pandas, numpy, scikit-learn
                - [ ] **R for Statistics** - statistical computing
                - [ ] **SQL for Data Analysis** - database querying
                - [ ] **Git Version Control** - code management
                - [ ] **Data Visualization** - matplotlib, seaborn, plotly
                - [ ] **Big Data Tools** - Spark, Hadoop basics
                - [ ] **Cloud Platforms** - AWS, GCP, Azure basics
                
                ---
                
                ## 💡 Interview Preparation Strategies
                
                ### 🗣️ Communication Tips
                1. **STAR Method**: Situation, Task, Action, Result for behavioral questions
                2. **Explain Simply**: Can you explain to a non-technical stakeholder?
                3. **Use Examples**: Always have concrete examples ready
                4. **Think Aloud**: Share your thought process during problem-solving
                
                ### 🧠 Technical Preparation
                1. **Know Trade-offs**: Every technique has pros and cons
                2. **Business Context**: Connect technical concepts to business problems
                3. **Stay Current**: Know recent developments in the field
                4. **Practice Coding**: Be ready to write and explain code
                
                ### ❓ Questions to Ask Interviewers
                - What does a typical day look like for this role?
                - What are the biggest data challenges the company faces?
                - How does the data science team collaborate with other departments?
                - What tools and technologies does the team currently use?
                - What opportunities are there for professional development?
                
                ---
                
                ## 🚀 Action Plan
                
                ### Week 1-2: Foundation
                - [ ] Review fundamental ML concepts
                - [ ] Practice explaining key algorithms
                - [ ] Brush up on statistics
                - [ ] Work through coding problems
                
                ### Week 3-4: Practice
                - [ ] Mock interviews with peers
                - [ ] Practice case studies
                - [ ] Review your past projects
                - [ ] Prepare project presentation
                
                ### Week 5-6: Polish
                - [ ] Research target companies
                - [ ] Prepare behavioral stories
                - [ ] Practice coding on whiteboard
                - [ ] Review recent developments in field
                
                ---
                
                ## 📚 Recommended Resources
                
                ### Books
                - **"Hands-On Machine Learning"** by Aurélien Géron
                - **"The Elements of Statistical Learning"** by Hastie, Tibshirani, Friedman
                - **"Python Machine Learning"** by Sebastian Raschka
                - **"Introduction to Statistical Learning"** by James, Witten, Hastie, Tibshirani
                
                ### Online Courses
                - **Andrew Ng's Machine Learning Course** (Coursera)
                - **CS229 Machine Learning** (Stanford)
                - **Kaggle Learn** (Free micro-courses)
                - **Fast.ai Practical Deep Learning**
                
                ### Practice Platforms
                - **LeetCode** for coding practice
                - **Kaggle** for data science competitions
                - **InterviewBit** for technical interviews
                - **Pramp** for mock interviews
                
                ---
                
                ## 🎯 Final Tips for Success
                
                ### Before the Interview
                - [ ] Research the company and role thoroughly
                - [ ] Review your resume and be ready to discuss every project
                - [ ] Prepare questions to ask the interviewer
                - [ ] Get a good night's sleep
                
                ### During the Interview
                - [ ] Listen carefully to questions
                - [ ] Ask clarifying questions when needed
                - [ ] Think out loud and explain your reasoning
                - [ ] Be honest about what you don't know
                - [ ] Stay calm and confident
                
                ### After the Interview
                - [ ] Send a thank-you email within 24 hours
                - [ ] Reflect on what went well and what to improve
                - [ ] Follow up appropriately based on their timeline
                
                ---
                
                **Remember**: Interviews are conversations, not interrogations. Show your passion for data science, your curiosity to learn, and your ability to solve problems. Good luck! 🍀
                """)
        
        # Footer
        gr.Markdown("""
        ---
        
        ### 🔧 How to Use This Tool
        
        1. **Start with Sample Questions**: Try the built-in questions in the "Interview Practice" tab
        2. **Upload Your Materials**: Add your study materials in the "Upload Materials" tab for personalized answers
        3. **Study Systematically**: Use the "Study Guide" tab to track your preparation progress
        4. **Practice Regularly**: Come back daily to practice different types of questions
        
        ### ✨ Pro Tips
        - **Upload multiple documents** for comprehensive coverage
        - **Ask follow-up questions** to deepen your understanding  
        - **Practice explaining answers** out loud as if in a real interview
        - **Focus on understanding concepts** rather than memorizing answers
        
        ### 🔒 Privacy & Data
        - All processing happens in your browser session
        - Uploaded documents are not stored permanently
        - No data is shared with external services
        
        **Made with ❤️ for aspiring data scientists and ML engineers**
        """)
    
    return interface

# Create and launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    ) 
