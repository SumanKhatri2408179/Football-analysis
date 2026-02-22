// import { StrictMode } from 'react'
// import { createRoot } from 'react-dom/client'
// import './index.css'
// import App from './App.jsx'

// createRoot(document.getElementById('root')).render(
//   <StrictMode>
//     <App />
//   </StrictMode>,
// )
// import React from "react";
// import ReactDOM from "react-dom/client";
// import App from "./App.jsx";
// import "./index.css";
// import { GoogleOAuthProvider } from "@react-oauth/google";

// const GOOGLE_CLIENT_ID = "YOUR_CLIENT_ID_HERE.apps.googleusercontent.com";

// ReactDOM.createRoot(document.getElementById("root")).render(
//   <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
//     <App />
//   </GoogleOAuthProvider>
// );
// import React from "react";
// import ReactDOM from "react-dom/client";
// import App from "./App.jsx";
// import "./index.css";
// import { GoogleOAuthProvider } from "@react-oauth/google";

// // Put your actual Google Client ID here
// const GOOGLE_CLIENT_ID = "863211431594-2ne6ga8t8jv1hendhf92302jhqr2565l.apps.googleusercontent.com";

// ReactDOM.createRoot(document.getElementById("root")).render(
//   <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
//     <App />
//   </GoogleOAuthProvider>
// );
// import React from "react";
// import ReactDOM from "react-dom/client";
// import App from "./App.jsx";
// import "./index.css";
// import { GoogleOAuthProvider } from "@react-oauth/google";
// import { AuthProvider } from "./contexts/AuthContext.jsx";

// ReactDOM.createRoot(document.getElementById("root")).render(
//   <GoogleOAuthProvider clientId="863211431594-2ne6ga8t8jv1hendhf92302jhqr2565l.apps.googleusercontent.com">
//     <AuthProvider>
//       <App />
//     </AuthProvider>
//   </GoogleOAuthProvider>
// );
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";
import { GoogleOAuthProvider } from "@react-oauth/google";
import { AuthProvider } from "./contexts/AuthContext.jsx";

// Your full Google OAuth Client ID
const GOOGLE_CLIENT_ID = "863211431594-2ne6ga8t8jv1hendhf92302jhqr2565l.apps.googleusercontent.com";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <AuthProvider>
        <App />
      </AuthProvider>
    </GoogleOAuthProvider>
  </React.StrictMode>
);
