// // import { useState } from 'react';
// // import { X, Mail, Lock, User } from 'lucide-react';

// // export default function LoginModal({ isOpen, onClose }) {
// //   const [isSignUp, setIsSignUp] = useState(false);
// //   const [formData, setFormData] = useState({
// //     name: '',
// //     email: '',
// //     password: ''
// //   });

// //   const handleSubmit = (e) => {
// //     e.preventDefault();
// //     console.log('Form submitted:', formData);
// //     // Add your authentication logic here
// //     onClose();
// //   };

// //   const handleChange = (e) => {
// //     setFormData({ ...formData, [e.target.name]: e.target.value });
// //   };

// //   if (!isOpen) return null;

// //   return (
// //     <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-in fade-in duration-200">
// //       <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl shadow-2xl max-w-md w-full p-8 relative border border-slate-700 animate-in slide-in-from-bottom-4 duration-300">
// //         {/* Close Button */}
// //         <button
// //           onClick={onClose}
// //           className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
// //         >
// //           <X size={24} />
// //         </button>

// //         {/* Header */}
// //         <div className="text-center mb-8">
// //           <h2 className="text-3xl font-bold text-white mb-2">
// //             {isSignUp ? 'Create Account' : 'Welcome Back'}
// //           </h2>
// //           <p className="text-gray-400">
// //             {isSignUp ? 'Sign up to continue' : 'Log in to continue'}
// //           </p>
// //         </div>

// //         {/* Form Fields */}
// //         <div className="space-y-5">
// //           {/* Name Field (Sign Up Only) */}
// //           {isSignUp && (
// //             <div className="relative">
// //               <User className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
// //               <input
// //                 type="text"
// //                 name="name"
// //                 value={formData.name}
// //                 onChange={handleChange}
// //                 placeholder="Full Name"
// //                 className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-11 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all"
// //               />
// //             </div>
// //           )}

// //           {/* Email Field */}
// //           <div className="relative">
// //             <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
// //             <input
// //               type="email"
// //               name="email"
// //               value={formData.email}
// //               onChange={handleChange}
// //               placeholder="Email Address"
// //               className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-11 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all"
// //             />
// //           </div>

// //           {/* Password Field */}
// //           <div className="relative">
// //             <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
// //             <input
// //               type="password"
// //               name="password"
// //               value={formData.password}
// //               onChange={handleChange}
// //               placeholder="Password"
// //               className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-11 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all"
// //             />
// //           </div>

// //           {/* Forgot Password (Login Only) */}
// //           {!isSignUp && (
// //             <div className="text-right">
// //               <button className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors">
// //                 Forgot Password?
// //               </button>
// //             </div>
// //           )}

// //           {/* Submit Button */}
// //           <button
// //             onClick={handleSubmit}
// //             className="w-full bg-gradient-to-r from-blue-500 to-cyan-400 text-white py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-cyan-500 transition-all transform hover:scale-105 shadow-lg"
// //           >
// //             {isSignUp ? 'Sign Up' : 'Log In'}
// //           </button>
// //         </div>

// //         {/* Toggle Sign Up/Login */}
// //         <div className="mt-6 text-center">
// //           <p className="text-gray-400">
// //             {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
// //             <button
// //               onClick={() => setIsSignUp(!isSignUp)}
// //               className="text-cyan-400 hover:text-cyan-300 font-semibold transition-colors"
// //             >
// //               {isSignUp ? 'Log In' : 'Sign Up'}
// //             </button>
// //           </p>
// //         </div>

// //         {/* Divider */}
// //         <div className="relative my-6">
// //           <div className="absolute inset-0 flex items-center">
// //             <div className="w-full border-t border-slate-600"></div>
// //           </div>
// //           <div className="relative flex justify-center text-sm">
// //             <span className="px-2 bg-slate-800 text-gray-400">Or continue with</span>
// //           </div>
// //         </div>

// //         {/* Social Login */}
// //         <div className="grid grid-cols-2 gap-3">
// //           <button className="flex items-center justify-center gap-2 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 text-white py-2.5 rounded-lg transition-all">
// //             <svg className="w-5 h-5" viewBox="0 0 24 24">
// //               <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
// //               <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
// //               <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
// //               <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
// //             </svg>
// //             Google
// //           </button>
// //           <button className="flex items-center justify-center gap-2 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 text-white py-2.5 rounded-lg transition-all">
// //             <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
// //               <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
// //             </svg>
// //             GitHub
// //           </button>
// //         </div>
// //       </div>
// //     </div>
// //   );
// // }
// import { useState } from 'react';
// import { X, Mail, Lock, User } from 'lucide-react';

// export default function LoginModal({ isOpen, onClose, onLoginSuccess }) {
//   const [isSignUp, setIsSignUp] = useState(false);
//   const [formData, setFormData] = useState({
//     name: '',
//     email: '',
//     password: ''
//   });

//   const handleSubmit = (e) => {
//     e.preventDefault();
//     console.log('Form submitted:', formData);
//     // Add your authentication logic here
    
//     // Call onLoginSuccess if provided
//     if (onLoginSuccess) {
//       onLoginSuccess();
//     } else {
//       onClose();
//     }
//   };

//   const handleChange = (e) => {
//     setFormData({ ...formData, [e.target.name]: e.target.value });
//   };

//   if (!isOpen) return null;

//   return (
//     <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-in fade-in duration-200">
//       <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl shadow-2xl max-w-md w-full p-8 relative border border-slate-700 animate-in slide-in-from-bottom-4 duration-300">
//         {/* Close Button */}
//         <button
//           onClick={onClose}
//           className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
//         >
//           <X size={24} />
//         </button>

//         {/* Header */}
//         <div className="text-center mb-8">
//           <h2 className="text-3xl font-bold text-white mb-2">
//             {isSignUp ? 'Create Account' : 'Welcome Back'}
//           </h2>
//           <p className="text-gray-400">
//             {isSignUp ? 'Sign up to continue' : 'Log in to continue'}
//           </p>
//         </div>

//         {/* Form Fields */}
//         <div className="space-y-5">
//           {/* Name Field (Sign Up Only) */}
//           {isSignUp && (
//             <div className="relative">
//               <User className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
//               <input
//                 type="text"
//                 name="name"
//                 value={formData.name}
//                 onChange={handleChange}
//                 placeholder="Full Name"
//                 className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-11 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all"
//               />
//             </div>
//           )}

//           {/* Email Field */}
//           <div className="relative">
//             <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
//             <input
//               type="email"
//               name="email"
//               value={formData.email}
//               onChange={handleChange}
//               placeholder="Email Address"
//               className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-11 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all"
//             />
//           </div>

//           {/* Password Field */}
//           <div className="relative">
//             <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
//             <input
//               type="password"
//               name="password"
//               value={formData.password}
//               onChange={handleChange}
//               placeholder="Password"
//               className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-11 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all"
//             />
//           </div>

//           {/* Forgot Password (Login Only) */}
//           {!isSignUp && (
//             <div className="text-right">
//               <button className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors">
//                 Forgot Password?
//               </button>
//             </div>
//           )}

//           {/* Submit Button */}
//           <button
//             onClick={handleSubmit}
//             className="w-full bg-gradient-to-r from-blue-500 to-cyan-400 text-white py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-cyan-500 transition-all transform hover:scale-105 shadow-lg"
//           >
//             {isSignUp ? 'Sign Up' : 'Log In'}
//           </button>
//         </div>

//         {/* Toggle Sign Up/Login */}
//         <div className="mt-6 text-center">
//           <p className="text-gray-400">
//             {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
//             <button
//               onClick={() => setIsSignUp(!isSignUp)}
//               className="text-cyan-400 hover:text-cyan-300 font-semibold transition-colors"
//             >
//               {isSignUp ? 'Log In' : 'Sign Up'}
//             </button>
//           </p>
//         </div>

//         {/* Divider */}
//         <div className="relative my-6">
//           <div className="absolute inset-0 flex items-center">
//             <div className="w-full border-t border-slate-600"></div>
//           </div>
//           <div className="relative flex justify-center text-sm">
//             <span className="px-2 bg-slate-800 text-gray-400">Or continue with</span>
//           </div>
//         </div>

//         {/* Social Login */}
//         <div className="grid grid-cols-2 gap-3">
//           <button className="flex items-center justify-center gap-2 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 text-white py-2.5 rounded-lg transition-all">
//             <svg className="w-5 h-5" viewBox="0 0 24 24">
//               <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
//               <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
//               <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
//               <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
//             </svg>
//             Google
//           </button>
//           <button className="flex items-center justify-center gap-2 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 text-white py-2.5 rounded-lg transition-all">
//             <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
//               <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
//             </svg>
//             GitHub
//           </button>
//         </div>
//       </div>
//     </div>
//   );
// }
// 
import { useState, useContext } from 'react';
import { X, Mail, Lock, User, Loader } from 'lucide-react';
import { useGoogleLogin } from '@react-oauth/google';
import axios from 'axios';
import { AuthContext } from '../contexts/AuthContext.jsx';

export default function LoginModal({ isOpen, onClose, onGoogleLoginSuccess }) {
  const { login, register, loginWithGoogle } = useContext(AuthContext);

  const [isSignUp, setIsSignUp] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: ''
  });

  // Google Login
  const googleLogin = useGoogleLogin({
    onSuccess: async (response) => {
      try {
        setLoading(true);
        setError('');

        const userInfo = await axios.get(
          'https://www.googleapis.com/oauth2/v3/userinfo',
          { headers: { Authorization: `Bearer ${response.access_token}` } }
        );

        const success = await loginWithGoogle({
          access_token: response.access_token,
          name: userInfo.data.name,
          email: userInfo.data.email,
          picture: userInfo.data.picture
        });

        if (success) {
          onGoogleLoginSuccess();
          onClose();
        }
      } catch (err) {
        console.error('Google login error:', err);
        setError('Google login failed. Please try again.');
      } finally {
        setLoading(false);
      }
    },
    onError: (err) => {
      console.error('Google login error:', err);
      setError('Google login failed. Please try again.');
    }
  });

  const handleSubmit = async () => {
    setLoading(true);
    setError('');

    try {
      if (!formData.email || !formData.password || (isSignUp && !formData.name)) {
        setError('Please fill in all required fields.');
        setLoading(false);
        return;
      }

      let success = false;
      if (isSignUp) {
        success = await register(formData.name, formData.email, formData.password);
      } else {
        success = await login(formData.email, formData.password);
      }

      if (success) {
        onClose();
      } else {
        setError('Authentication failed. Please check your credentials.');
      }
    } catch (err) {
      console.error('Login/Register error:', err);
      setError('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl shadow-2xl max-w-md w-full p-8 relative border border-slate-700">
        <button onClick={onClose} className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors">
          <X size={24} />
        </button>

        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">{isSignUp ? 'Create Account' : 'Welcome Back'}</h2>
          <p className="text-gray-400">{isSignUp ? 'Sign up to continue' : 'Log in to continue'}</p>
        </div>

        {error && (
          <div className="bg-red-500/20 border border-red-500 text-red-400 px-4 py-2 rounded-lg mb-4 text-sm text-center">
            {error}
          </div>
        )}

        <div className="space-y-5">
          {isSignUp && (
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
              <input type="text" name="name" value={formData.name} onChange={handleChange} placeholder="Full Name"
                disabled={loading} className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-11 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all disabled:opacity-50" />
            </div>
          )}

          <div className="relative">
            <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
            <input type="email" name="email" value={formData.email} onChange={handleChange} placeholder="Email Address"
              disabled={loading} className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-11 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all disabled:opacity-50" />
          </div>

          <div className="relative">
            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
            <input type="password" name="password" value={formData.password} onChange={handleChange} placeholder="Password"
              disabled={loading} className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-11 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all disabled:opacity-50" />
          </div>

          {!isSignUp && <div className="text-right"><button className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors">Forgot Password?</button></div>}

          <button onClick={handleSubmit} disabled={loading}
            className="w-full bg-gradient-to-r from-blue-500 to-cyan-400 text-white py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-cyan-500 transition-all transform hover:scale-105 shadow-lg disabled:opacity-50 flex items-center justify-center gap-2">
            {loading && <Loader size={18} className="animate-spin" />}
            {loading ? (isSignUp ? 'Signing Up...' : 'Logging In...') : (isSignUp ? 'Sign Up' : 'Log In')}
          </button>
        </div>

        <div className="mt-6 text-center">
          <p className="text-gray-400">{isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
            <button onClick={() => { setIsSignUp(!isSignUp); setError(''); }}
              className="text-cyan-400 hover:text-cyan-300 font-semibold transition-colors">{isSignUp ? 'Log In' : 'Sign Up'}</button>
          </p>
        </div>

        <div className="relative my-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-slate-600"></div>
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-slate-800 text-gray-400">Or continue with</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <button onClick={googleLogin} disabled={loading}
            className="flex items-center justify-center gap-2 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 text-white py-2.5 rounded-lg transition-all disabled:opacity-50">
            {loading ? <Loader size={18} className="animate-spin" /> : 'Google'}
          </button>

          <button className="flex items-center justify-center gap-2 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 text-white py-2.5 rounded-lg transition-all">
            GitHub
          </button>
        </div>
      </div>
    </div>
  );
}
