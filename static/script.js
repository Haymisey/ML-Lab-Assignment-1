// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const form = document.getElementById('predictionForm');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const loadingOverlay = document.getElementById('loadingOverlay');
const buttons = document.querySelectorAll('.btn');

// Event Listeners
form.addEventListener('submit', (e) => {
    e.preventDefault();
    setActiveButton('both');
    makePrediction('both');
});

buttons.forEach(btn => {
    btn.addEventListener('click', () => {
        const model = btn.dataset.model;
        setActiveButton(model);
        if (btn.type === 'button') {
            makePrediction(model);
        }
    });
});

// Set active button styling
function setActiveButton(modelType) {
    buttons.forEach(btn => {
        if (btn.dataset.model === modelType) {
            btn.classList.remove('btn-secondary');
            btn.classList.add('btn-primary', 'active');
        } else {
            btn.classList.remove('btn-primary', 'active');
            btn.classList.add('btn-secondary');
        }
    });
}

// Collect form data
function getFormData() {
    const formData = new FormData(form);
    return {
        pregnancies: parseInt(formData.get('pregnancies')),
        glucose: parseFloat(formData.get('glucose')),
        blood_pressure: parseFloat(formData.get('blood_pressure')),
        skin_thickness: parseFloat(formData.get('skin_thickness')),
        insulin: parseFloat(formData.get('insulin')),
        bmi: parseFloat(formData.get('bmi')),
        diabetes_pedigree: parseFloat(formData.get('diabetes_pedigree')),
        age: parseInt(formData.get('age'))
    };
}

// Make prediction
async function makePrediction(modelType) {
    const data = getFormData();
    
    // Show loading
    loadingOverlay.style.display = 'flex';
    resultsSection.style.display = 'none';
    
    try {
        let results = [];
        
        if (modelType === 'both') {
            // Call both endpoints
            const [dtResult, lrResult] = await Promise.allSettled([
                fetch(`${API_BASE_URL}/predict/decision-tree`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                }),
                fetch(`${API_BASE_URL}/predict/logistic-regression`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                })
            ]);
            
            if (dtResult.status === 'fulfilled' && dtResult.value.ok) {
                results.push(await dtResult.value.json());
            }
            if (lrResult.status === 'fulfilled' && lrResult.value.ok) {
                results.push(await lrResult.value.json());
            }
        } else {
            // Call single endpoint
            const endpoint = modelType === 'decision-tree' 
                ? '/predict/decision-tree' 
                : '/predict/logistic-regression';
            
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            if (response.ok) {
                results.push(await response.json());
            } else {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        }
        
        if (results.length === 0) {
            throw new Error('No predictions received');
        }
        
        displayResults(results);
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message);
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

// Display results
function displayResults(results) {
    resultsContainer.innerHTML = '';
    
    results.forEach(result => {
        const card = createResultCard(result);
        resultsContainer.appendChild(card);
    });
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Create result card
function createResultCard(result) {
    const card = document.createElement('div');
    const isDiabetic = result.prediction === 1;
    const statusClass = isDiabetic ? 'diabetic' : 'non-diabetic';
    
    card.className = `result-card ${statusClass}`;
    
    let probabilityHTML = '';
    if (result.probability !== null && result.probability !== undefined) {
        const percentage = (result.probability * 100).toFixed(1);
        probabilityHTML = `
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${percentage}%"></div>
            </div>
            <div class="probability-text">
                Diabetes Risk: ${percentage}%
            </div>
        `;
    }
    
    card.innerHTML = `
        <div class="result-header">
            <h3 class="result-model">${result.model_type}</h3>
        </div>
        <div class="result-prediction">
            <div class="prediction-label ${statusClass}">
                ${result.prediction_label}
            </div>
            ${probabilityHTML}
        </div>
    `;
    
    return card;
}

// Show error
function showError(message) {
    resultsContainer.innerHTML = `
        <div class="result-card" style="border-color: var(--danger-color);">
            <div class="result-header">
                <h3 class="result-model">Error</h3>
            </div>
            <p style="color: var(--text-secondary); text-align: center;">
                ${message}
            </p>
        </div>
    `;
    resultsSection.style.display = 'block';
}

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const health = await response.json();
        console.log('API Health:', health);
        
        if (!health.models_loaded.decision_tree && !health.models_loaded.logistic_regression) {
            console.warn('No models loaded! Please add model files to the models/ directory.');
        }
    } catch (error) {
        console.error('API health check failed:', error);
    }
});
