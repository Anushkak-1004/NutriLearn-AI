import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import HomePage from './pages/HomePage';
import AnalyzePage from './pages/AnalyzePage';
import DashboardPage from './pages/DashboardPage';
import LearningPage from './pages/LearningPage';
import MLOpsDashboard from './pages/MLOpsDashboard';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen">
        <Navigation />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/learning/:moduleId" element={<LearningPage />} />
          <Route path="/mlops" element={<MLOpsDashboard />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
