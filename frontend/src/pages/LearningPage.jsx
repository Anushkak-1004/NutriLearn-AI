import { useState, useEffect } from 'react';
import { useNavigate, useLocation, useParams } from 'react-router-dom';
import { ArrowLeft, BookOpen, Award, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { completeModule } from '../utils/api';
import { getUserId } from '../utils/storage';

function LearningPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { moduleId } = useParams();
  const [module, setModule] = useState(null);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [score, setScore] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [pointsEarned, setPointsEarned] = useState(0);

  useEffect(() => {
    // Get module from location state or fetch from API
    if (location.state?.module) {
      setModule(location.state.module);
    } else {
      // In a real app, fetch module data from API
      console.error('Module data not found');
      navigate('/dashboard');
    }
  }, [location, navigate]);

  const handleAnswerSelect = (questionIndex, answerIndex) => {
    setSelectedAnswers({
      ...selectedAnswers,
      [questionIndex]: answerIndex,
    });
  };

  const handleSubmitQuiz = async () => {
    if (!module) return;

    // Calculate score
    const questions = module.quiz.questions;
    let correct = 0;

    questions.forEach((question, index) => {
      if (selectedAnswers[index] === question.correct) {
        correct++;
      }
    });

    const percentage = Math.round((correct / questions.length) * 100);
    setScore(percentage);
    setShowResults(true);

    // Submit to backend if score >= 70%
    if (percentage >= 70) {
      setSubmitting(true);
      try {
        const userId = getUserId();
        const result = await completeModule(userId, moduleId, percentage);
        setPointsEarned(result.points_earned);
        setCompleted(true);
      } catch (error) {
        console.error('Error completing module:', error);
      } finally {
        setSubmitting(false);
      }
    }
  };

  if (!module) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-emerald-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading module...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <button
          onClick={() => navigate('/dashboard')}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 mb-6 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span className="font-medium">Back to Dashboard</span>
        </button>

        {/* Module Header */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="bg-emerald-100 p-3 rounded-xl">
                <BookOpen className="w-8 h-8 text-emerald-600" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-800">{module.title}</h1>
                <p className="text-gray-600 mt-1">{module.reason}</p>
              </div>
            </div>
            <div className="bg-gradient-to-r from-amber-400 to-orange-500 text-white px-4 py-2 rounded-full font-bold flex items-center space-x-2">
              <Award className="w-5 h-5" />
              <span>{module.points} pts</span>
            </div>
          </div>
          <div className="text-sm text-gray-500">
            ⏱ Estimated time: {module.estimated_time} minutes
          </div>
        </div>

        {/* Content Sections */}
        <div className="space-y-6 mb-8">
          {module.content.map((section, index) => (
            <div key={index} className="bg-white rounded-2xl shadow-lg p-6">
              {section.type === 'text' && (
                <p className="text-gray-700 leading-relaxed">{section.data}</p>
              )}

              {section.type === 'infographic' && (
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-xl">
                  <p className="text-gray-800 font-medium">{section.data}</p>
                </div>
              )}

              {section.type === 'list' && (
                <div>
                  <p className="font-bold text-gray-800 mb-3">Good Sources:</p>
                  <ul className="space-y-2">
                    {section.data.map((item, i) => (
                      <li key={i} className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-emerald-600 rounded-full"></div>
                        <span className="text-gray-700">{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {section.type === 'comparison' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-green-50 p-4 rounded-xl">
                    <p className="font-bold text-green-700 mb-2">✓ Choose These:</p>
                    <ul className="space-y-1">
                      {section.data.good.map((item, i) => (
                        <li key={i} className="text-gray-700">• {item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="bg-red-50 p-4 rounded-xl">
                    <p className="font-bold text-red-700 mb-2">✗ Limit These:</p>
                    <ul className="space-y-1">
                      {section.data.limit.map((item, i) => (
                        <li key={i} className="text-gray-700">• {item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {section.type === 'tips' && (
                <div>
                  <p className="font-bold text-gray-800 mb-3">💡 Tips:</p>
                  <div className="space-y-2">
                    {section.data.map((tip, i) => (
                      <div key={i} className="bg-blue-50 p-3 rounded-lg text-gray-700">
                        {tip}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {section.type === 'schedule' && (
                <div>
                  <p className="font-bold text-gray-800 mb-3">Recommended Timing:</p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Object.entries(section.data).map(([meal, time]) => (
                      <div key={meal} className="bg-gradient-to-br from-purple-50 to-pink-50 p-4 rounded-xl text-center">
                        <p className="font-bold text-gray-800 capitalize">{meal}</p>
                        <p className="text-gray-600">{time}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Quiz Section */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Knowledge Check</h2>

          <div className="space-y-6">
            {module.quiz.questions.map((question, qIndex) => (
              <div key={qIndex} className="border-b border-gray-200 pb-6 last:border-0">
                <p className="font-medium text-gray-800 mb-4">
                  {qIndex + 1}. {question.question}
                </p>

                <div className="space-y-2">
                  {question.options.map((option, oIndex) => {
                    const isSelected = selectedAnswers[qIndex] === oIndex;
                    const isCorrect = question.correct === oIndex;
                    const showFeedback = showResults;

                    return (
                      <button
                        key={oIndex}
                        onClick={() => !showResults && handleAnswerSelect(qIndex, oIndex)}
                        disabled={showResults}
                        className={`w-full text-left p-4 rounded-xl border-2 transition-all ${
                          showFeedback
                            ? isCorrect
                              ? 'border-green-500 bg-green-50'
                              : isSelected
                              ? 'border-red-500 bg-red-50'
                              : 'border-gray-200 bg-gray-50'
                            : isSelected
                            ? 'border-emerald-500 bg-emerald-50'
                            : 'border-gray-200 hover:border-gray-300'
                        } ${showResults ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-gray-800">{option}</span>
                          {showFeedback && isCorrect && (
                            <CheckCircle className="w-5 h-5 text-green-600" />
                          )}
                          {showFeedback && isSelected && !isCorrect && (
                            <XCircle className="w-5 h-5 text-red-600" />
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {!showResults && (
            <button
              onClick={handleSubmitQuiz}
              disabled={Object.keys(selectedAnswers).length !== module.quiz.questions.length}
              className="w-full mt-6 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-3 px-6 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Submit Quiz
            </button>
          )}

          {showResults && (
            <div className="mt-6">
              <div className={`p-6 rounded-xl ${
                score >= 70 ? 'bg-green-50 border-2 border-green-500' : 'bg-red-50 border-2 border-red-500'
              }`}>
                <div className="text-center">
                  <p className="text-2xl font-bold text-gray-800 mb-2">
                    Your Score: {score}%
                  </p>
                  {score >= 70 ? (
                    <>
                      <p className="text-green-700 mb-4">Great job! You passed the quiz!</p>
                      {submitting ? (
                        <div className="flex items-center justify-center space-x-2">
                          <Loader2 className="w-5 h-5 animate-spin" />
                          <span>Saving progress...</span>
                        </div>
                      ) : completed ? (
                        <div className="bg-white p-4 rounded-xl">
                          <div className="flex items-center justify-center space-x-2 text-amber-600 mb-2">
                            <Award className="w-6 h-6" />
                            <span className="text-xl font-bold">+{pointsEarned} Points Earned!</span>
                          </div>
                          <button
                            onClick={() => navigate('/dashboard')}
                            className="mt-4 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-6 rounded-xl transition-colors"
                          >
                            Continue Learning
                          </button>
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <>
                      <p className="text-red-700 mb-4">You need 70% or higher to pass. Try again!</p>
                      <button
                        onClick={() => {
                          setShowResults(false);
                          setSelectedAnswers({});
                          setScore(0);
                        }}
                        className="bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-6 rounded-xl transition-colors"
                      >
                        Retry Quiz
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default LearningPage;
