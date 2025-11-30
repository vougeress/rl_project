import { useState } from 'react';
import { apiService } from '../../services/api';
import { UserRegistration } from '../../types';
import { Check } from 'lucide-react';
import './Login.css';

interface LoginProps {
  onLogin: (userId: number) => void;
}

export function Login({ onLogin }: LoginProps) {
  const [currentStep, setCurrentStep] = useState(1);
  type RegistrationForm = UserRegistration & { gender: 'male' | 'female' | null };
  const [formData, setFormData] = useState<RegistrationForm>({
    name: '',
    age: 18,
    gender: null
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const steps = [
    { number: 1, label: 'Name' },
    { number: 2, label: 'Age' },
    { number: 3, label: 'Gender' }
  ];

  const handleNext = () => {
    setError(null);

    // Validation for each step
    if (currentStep === 1 && !formData.name.trim()) {
      setError('Please enter your name');
      return;
    }

    if (currentStep === 2) {
      if (!formData.age || formData.age < 16 || formData.age > 100) {
        setError('Please enter a valid age (16-100)');
        return;
      }
    }

    if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.gender) {
      setError('Please select a gender to continue');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const payload: UserRegistration = {
        name: formData.name,
        age: formData.age
      };
      const response = await apiService.registerUser(payload);
      onLogin(response.user_id);
    } catch (err) {
      setError('Failed to register. Please try again.');
      console.error('Login error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="step-content">
            <div className="form-group">
              <label htmlFor="name" className="form-label">
                What's your name?
              </label>
              <input
                id="name"
                type="text"
                className="form-input"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="Enter your name"
                autoFocus
                onKeyPress={(e) => e.key === 'Enter' && handleNext()}
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className="step-content">
            <div className="form-group">
              <label htmlFor="age" className="form-label">
                How old are you?
              </label>
              <input
                id="age"
                type="number"
                className="form-input"
                value={formData.age}
                onChange={(e) => setFormData({ ...formData, age: parseInt(e.target.value) || 0 })}
                placeholder="Enter your age"
                min={16}
                max={100}
                autoFocus
                onKeyPress={(e) => e.key === 'Enter' && handleNext()}
              />
            </div>
          </div>
        );

      case 3:
        return (
          <div className="step-content">
            <div className="form-group">
              <label className="form-label">Select your gender</label>
              <div className="gender-options">
                <div
                  className={`gender-option ${formData.gender === 'male' ? 'selected' : ''}`}
                  onClick={() => setFormData({ ...formData, gender: 'male' })}
                >
                  <div className="gender-icon">👨</div>
                  <div className="gender-label">Male</div>
                </div>

                <div
                  className={`gender-option ${formData.gender === 'female' ? 'selected' : ''}`}
                  onClick={() => setFormData({ ...formData, gender: 'female' })}
                >
                  <div className="gender-icon">👩</div>
                  <div className="gender-label">Female</div>
                </div>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="progress-indicator">
          {steps.map((step, index) => (
            <div key={step.number} style={{ display: 'contents' }}>
              <div className="progress-step">
                <div
                  className={`step-circle ${
                    currentStep === step.number
                      ? 'active'
                      : currentStep > step.number
                      ? 'completed'
                      : ''
                  }`}
                >
                  {currentStep > step.number ? (
                    <Check size={24} />
                  ) : (
                    <span>{step.number}</span>
                  )}
                </div>
                <div
                  className={`step-label ${
                    currentStep === step.number
                      ? 'active'
                      : currentStep > step.number
                      ? 'completed'
                      : ''
                  }`}
                >
                  {step.label}
                </div>
              </div>
              {index < steps.length - 1 && (
                <div
                  className={`step-connector ${
                    currentStep > step.number ? 'completed' : ''
                  }`}
                />
              )}
            </div>
          ))}
        </div>

        <h1 className="login-title">Welcome</h1>
        <p className="login-subtitle">
          {currentStep === 1 && "Let's start with your name"}
          {currentStep === 2 && 'How old are you?'}
          {currentStep === 3 && 'Almost done! Select your gender'}
        </p>

        <form onSubmit={handleSubmit} className="login-form">
          {renderStepContent()}

          {error && <div className="error-message">{error}</div>}

          <div className="form-actions">
            {currentStep > 1 && (
              <button
                type="button"
                className="back-button-login"
                onClick={handleBack}
              >
                Back
              </button>
            )}

            {currentStep < 3 ? (
              <button
                type="button"
                className="login-button"
                onClick={handleNext}
              >
                Next
              </button>
            ) : (
              <button
                type="submit"
                className="login-button"
                disabled={isLoading || !formData.gender}
              >
                {isLoading ? 'Loading...' : 'Get Started'}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
