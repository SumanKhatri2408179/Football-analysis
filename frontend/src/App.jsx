import { useState, useContext } from "react";
import Demo from "./section/Demo";
import Faq from "./section/Faq";
import Features from "./section/Features";
import Footer from "./section/Footer";
import Header from "./section/Header";
import Hero from "./section/Hero";
import Slider from "./section/Slider";
import VideoUploader from "./section/VideoUploader";
import LoginModal from "./components/LoginModal"; // Import your login modal
import { AuthContext } from "./contexts/AuthContext"; // Import AuthContext

const App = () => {
  const { isAuthenticated, user, logout } = useContext(AuthContext);
  const [isLoginOpen, setIsLoginOpen] = useState(false);

  return (
    <main className="overflow-hidden">
      {/* Header */}
      <Header />

      {/* Optional login/logout button at top-right or somewhere */}
      <div className="fixed top-4 right-4 z-50">
        {!isAuthenticated ? (
          <button
            onClick={() => setIsLoginOpen(true)}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition"
          >
            Login / Sign Up
          </button>
        ) : (
          <div className="flex items-center gap-3">
            <span className="text-white">Hi, {user?.name}</span>
            <button
              onClick={logout}
              className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition"
            >
              Logout
            </button>
          </div>
        )}
      </div>

      {/* Login Modal */}
      <LoginModal
        isOpen={isLoginOpen}
        onClose={() => setIsLoginOpen(false)}
        onLoginSuccess={() => setIsLoginOpen(false)}
      />

      {/* Main Sections */}
      <Hero />
      <Features />
      <VideoUploader />
      <Slider />
      <Faq />
      <Demo />
      <Footer />
    </main>
  );
};

export default App;
