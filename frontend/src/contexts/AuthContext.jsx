// import { createContext, useState, useEffect } from "react";
// import { loginUser, fetchUserProfile, registerUser } from "../api";
// import PropTypes from "prop-types";

// const AuthContext = createContext({});

// const AuthProvider = ({ children }) => {
//   const [token, setToken] = useState(localStorage.getItem("token"));
//   const [user, setUser] = useState(null);
//   const isAuthenticated = !!token;

//   useEffect(() => {
//     if (token) {
//       const getUser = async () => {
//         try {
//           const userProfile = await fetchUserProfile(token);
//           setUser(userProfile);
//         } catch (err) {
//           logout(); // token expired / invalid
//         }
//       };
//       getUser();
//     }
//   }, [token]);

//   const login = async (username, password) => {
//     const response = await loginUser({ username, password });

//     if (response?.access_token) {
//       setToken(response.access_token);
//       localStorage.setItem("token", response.access_token);

//       const userProfile = await fetchUserProfile(response.access_token);
//       setUser(userProfile);

//       return true; // ✅ important for modal close
//     }
//     return false;
//   };

//   const register = async (username, email, password) => {
//     await registerUser({ username, email, password });
//   };

//   const logout = () => {
//     setToken(null);
//     setUser(null);
//     localStorage.removeItem("token");
//   };

//   return (
//     <AuthContext.Provider
//       value={{
//         token,
//         user,
//         isAuthenticated,
//         login,
//         register,
//         logout,
//       }}
//     >
//       {children}
//     </AuthContext.Provider>
//   );
// };

// AuthProvider.propTypes = {
//   children: PropTypes.node.isRequired,
// };

// export { AuthProvider, AuthContext };
import { createContext, useState, useEffect } from "react";
import { loginUser, fetchUserProfile, registerUser } from "../api";
import PropTypes from "prop-types";

const AuthContext = createContext({});

const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [user, setUser] = useState(() => {
    const savedUser = localStorage.getItem("user");
    return savedUser ? JSON.parse(savedUser) : null;
  });
  const isAuthenticated = !!token;

  // Fetch user profile if token exists and user not loaded
  useEffect(() => {
    if (token && !user) {
      const getUser = async () => {
        try {
          const userProfile = await fetchUserProfile(token);
          setUser(userProfile);
          localStorage.setItem("user", JSON.stringify(userProfile));
        } catch (err) {
          console.error("Failed to fetch user profile:", err);
          logout(); // token expired / invalid
        }
      };
      getUser();
    }
  }, [token, user]);

  // Email/Password login
  const login = async (username, password) => {
    try {
      const response = await loginUser({ username, password });

      if (response?.access_token) {
        setToken(response.access_token);
        localStorage.setItem("token", response.access_token);

        const userProfile = await fetchUserProfile(response.access_token);
        setUser(userProfile);
        localStorage.setItem("user", JSON.stringify(userProfile));

        return true; // for modal close
      }
    } catch (err) {
      console.error("Login error:", err);
    }
    return false;
  };

  // Google login
  const loginWithGoogle = async (googleUser) => {
    if (!googleUser) return false;

    try {
      // Save token and user info from Google
      setToken(googleUser.access_token);
      localStorage.setItem("token", googleUser.access_token);

      const userProfile = {
        name: googleUser.name,
        email: googleUser.email,
        picture: googleUser.picture,
      };
      setUser(userProfile);
      localStorage.setItem("user", JSON.stringify(userProfile));

      return true;
    } catch (err) {
      console.error("Google login error:", err);
      return false;
    }
  };

  // Signup / register
  const register = async (username, email, password) => {
    try {
      await registerUser({ username, email, password });
      // Optional: auto login after signup
      return await login(email, password);
    } catch (err) {
      console.error("Register error:", err);
      return false;
    }
  };

  // Logout
  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("token");
    localStorage.removeItem("user");
  };

  return (
    <AuthContext.Provider
      value={{
        token,
        user,
        isAuthenticated,
        login,
        loginWithGoogle,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

AuthProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export { AuthProvider, AuthContext };
