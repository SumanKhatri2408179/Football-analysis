// // import { useState } from "react";
// // import axios from "axios";
// // import Dropzone from "react-dropzone";
// // import "dropzone/dist/dropzone.css"; // Import Dropzone CSS for styling
// // import Button from "../components/Button";
// // import { Element } from "react-scroll";
// // import { API_URL } from "../api";
// // import { Copy } from "lucide-react";

// // import {
// //   Dialog,
// //   DialogContent,
// //   DialogDescription,
// //   DialogFooter,
// //   DialogHeader,
// //   DialogTitle,
// //   DialogTrigger,
// // } from "@/components/ui/dialog";
// // import { Input } from "@/components/ui/input";
// // import { Label } from "@/components/ui/label";

// // const VideoUploader = () => {
// //   const [videoFile, setVideoFile] = useState(null);
// //   const [outputVideo, setOutputVideo] = useState(null);
// //   const [videoApi, setVideoApi] = useState(null);
// //   const [progress, setProgress] = useState(0);

// //   const buttonIcon = {
// //     src: "/images/plan-1.png",
// //     alt: "button Logo",
// //   };

// //   const videoViewIcon = {
// //     src: "/images/plan-2.png",
// //     alt: "view video logo",
// //   };

// //   const handleCopyLink = () => {
// //     navigator.clipboard.writeText(videoApi);
// //     alert("Link copied to clipboard!");
// //   };

// //   // Handle file drop
// //   const handleDrop = (acceptedFiles) => {
// //     setVideoFile(acceptedFiles[0]); // Save the first dropped file
// //   };

// //   // Handle file upload
// //   const handleUpload = async () => {
// //     if (!videoFile) return;

// //     const formData = new FormData();
// //     formData.append("file", videoFile);

// //     try {
// //       const uploadResponse = await axios.post(
// //         `${API_URL}/upload-video/`,
// //         formData,
// //         {
// //           headers: { "Content-Type": "multipart/form-data" },
// //           onUploadProgress: (event) => {
// //             const percentCompleted = Math.round(
// //               (event.loaded * 100) / event.total
// //             );
// //             setProgress(percentCompleted);
// //           },
// //         }
// //       );
// //       setOutputVideo(uploadResponse.data.video_url);
// //       console.log("Video uploaded:", uploadResponse.data);
// //       console.log("upLoad video :", outputVideo);
// //     } catch (error) {
// //       console.error("Error uploading video", error);
// //       alert("Failed to upload the video. Please try again.");
// //     } finally {
// //       setProgress(0);
// //     }
// //   };

// //   const handleStream = () => {
// //     {
// //       outputVideo && setVideoApi(`${API_URL}/output/${outputVideo}`);
// //     }
// //     console.log(videoApi);
// //     // setVideoApi("http://localhost:8000/output/fixed_video.mp4");
// //   };

// //   return (
// //     <section>
// //       <Element name="create">
// //         <div className="container">
// //           <div className="flex flex-col items-center px-0 py-20 space-y-4 m-10 border-8 border-double border-[#34477C] rounded-2xl max-w-fit mx-auto ">
// //             <div className="flex flex-col gap-4 items-center justify-center mx-[190px]">
// //               <div className="w-auto">
// //                 <h1 className="mb-2 h1 text-p4 uppercase max-lg:mb-2 max-lg:h2 max-md:mb-2 max-md:text-5xl max-md:leading-12">
// //                   Make it happen!
// //                 </h1>
// //                 <p className="max-w-fit mb-6 body-1 max-md:mb-10 mx-auto">
// //                   ⚽Get your own video and realize its analytics.⚽
// //                 </p>
// //                 <div className="w-512 mx-auto">
// //                   <Dropzone onDrop={handleDrop} accept={{ "video/*": [] }}>
// //                     {({ getRootProps, getInputProps }) => (
// //                       <div
// //                         {...getRootProps()}
// //                         className="border-4 border-dashed border-blue-500 p-6 rounded-md text-center cursor-pointer hover:bg-blue-50"
// //                       >
// //                         <input {...getInputProps()} />
// //                         <p className="text-gray-600">
// //                           Drag & Drop a video file here, or click to select
// //                         </p>
// //                       </div>
// //                     )}
// //                   </Dropzone>
// //                 </div>
// //               </div>

// //               <Button
// //                 onClick={handleUpload}
// //                 disabled={!videoFile}
// //                 icon={buttonIcon.src}
// //               >
// //                 Upload and Process
// //               </Button>

// //               {/* Progress Bar */}
// //               {progress > 0 && (
// //                 <div className="w-[30%] text-center">
// //                   <p className="text-gray-600 mb-1">
// //                     {" "}
// //                     {progress === 100
// //                       ? "Video uploaded, wait for response..."
// //                       : `Uploading ${progress}%...`}
// //                   </p>
// //                   {progress !== 100 && (
// //                     <div className="w-full bg-gray-300 rounded-full h-4">
// //                       <div
// //                         className="bg-blue-500 h-4 rounded-full transition-all duration-200 ease-in-out"
// //                         style={{ width: `${progress}%` }}
// //                       ></div>
// //                     </div>
// //                   )}
// //                 </div>
// //               )}

// //               {outputVideo && (
// //                 <Dialog>
// //                   <DialogTrigger asChild>
// //                     <Button icon={videoViewIcon.src} onClick={handleStream}>
// //                       View Processed video
// //                     </Button>
// //                   </DialogTrigger>

// //                   <DialogContent className="shad-dialog g7">
// //                     <DialogHeader>
// //                       <DialogTitle>Football Analytics Video</DialogTitle>
// //                       <DialogDescription>
// //                         You can share this with anyone who has this link.
// //                       </DialogDescription>
// //                     </DialogHeader>
// //                     <video
// //                       name="outputVideo"
// //                       width={855}
// //                       height={655}
// //                       className="rounded-xl"
// //                       autoPlay
// //                       muted
// //                       loop
// //                       controls
// //                       onError={(e) => {
// //                         alert("Failed to load the video.");
// //                         console.log("Error e.target.error =", e.target.error);
// //                         console.log(videoApi);
// //                       }}
// //                     >
// //                       <source id="video-source" src={videoApi} type="video/mp4" />
// //                     </video>

// //                     <div className="flex items-center space-x-2">
// //                       <div className="grid flex-1 gap-2">
// //                         <Label htmlFor="link" className="sr-only">
// //                           Link
// //                         </Label>
// //                         <Input id="link" defaultValue={videoApi} readOnly className="g7" />
// //                       </div>
// //                       <button
// //                         type="submit"
// //                         size="sm"
// //                         className="px-3"
// //                         aria-label="Copy link to clipboard"
// //                         onClick={handleCopyLink}
// //                       >
// //                         <span className="sr-only">Copy</span>
// //                         <Copy />
// //                       </button>
// //                     </div>
// //                     <DialogFooter className="sm:justify-start">
// //                       <button
// //                         onClick={() => {
// //                           if (outputVideo) {
// //                             const filename = videoApi.split("/").pop() || "processed_test.mp4";
// //                             const downloadLink = `${API_URL}/download/${filename}`;
// //                             console.log(filename);
// //                             console.log(downloadLink);

// //                             window.open(downloadLink, "_blank").focus;
// //                           } else {
// //                             alert("No video available for download.");
// //                           }
// //                         }}
// //                         className="px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
// //                       >
// //                         DOWNLOAD
// //                       </button>
// //                     </DialogFooter>
// //                   </DialogContent>
// //                 </Dialog>
// //               )}
// //               {/* space */}
// //               <p className="text-sm text-gray-500">
// //                 Video processing may take up to 2 minutes. Feel free to leave
// //                 this page and come back later!
// //               </p>
// //               {/* space */}
// //             </div>
// //           </div>
// //         </div>
// //       </Element>
// //     </section>
// //   );
// // };

// // export default VideoUploader;
// import { useState } from "react";
// import axios from "axios";
// import Dropzone from "react-dropzone";
// import "dropzone/dist/dropzone.css";
// import Button from "../components/Button";
// import { Element } from "react-scroll";
// import { API_URL } from "../api";
// import { Copy } from "lucide-react";
// import LoginModal from "../components/LoginModal";

// import {
//   Dialog,
//   DialogContent,
//   DialogDescription,
//   DialogFooter,
//   DialogHeader,
//   DialogTitle,
//   DialogTrigger,
// } from "@/components/ui/dialog";
// import { Input } from "@/components/ui/input";
// import { Label } from "@/components/ui/label";

// const VideoUploader = () => {
//   const [videoFile, setVideoFile] = useState(null);
//   const [outputVideo, setOutputVideo] = useState(null);
//   const [videoApi, setVideoApi] = useState(null);
//   const [progress, setProgress] = useState(0);
//   const [showLogin, setShowLogin] = useState(false);
//   const [isAuthenticated, setIsAuthenticated] = useState(false);

//   const buttonIcon = {
//     src: "/images/plan-1.png",
//     alt: "button Logo",
//   };

//   const videoViewIcon = {
//     src: "/images/plan-2.png",
//     alt: "view video logo",
//   };

//   const handleCopyLink = () => {
//     navigator.clipboard.writeText(videoApi);
//     alert("Link copied to clipboard!");
//   };

//   // Handle file drop
//   const handleDrop = (acceptedFiles) => {
//     setVideoFile(acceptedFiles[0]);
//   };

//   // Handle upload button click
//   const handleUploadClick = () => {
//     if (!isAuthenticated) {
//       setShowLogin(true);
//       return;
//     }
//     handleUpload();
//   };

//   // Handle file upload
//   const handleUpload = async () => {
//     if (!videoFile) return;

//     const formData = new FormData();
//     formData.append("file", videoFile);

//     try {
//       const uploadResponse = await axios.post(
//         `${API_URL}/upload-video/`,
//         formData,
//         {
//           headers: { "Content-Type": "multipart/form-data" },
//           onUploadProgress: (event) => {
//             const percentCompleted = Math.round(
//               (event.loaded * 100) / event.total
//             );
//             setProgress(percentCompleted);
//           },
//         }
//       );
//       setOutputVideo(uploadResponse.data.video_url);
//       console.log("Video uploaded:", uploadResponse.data);
//       console.log("upLoad video :", outputVideo);
//     } catch (error) {
//       console.error("Error uploading video", error);
//       alert("Failed to upload the video. Please try again.");
//     } finally {
//       setProgress(0);
//     }
//   };

//   const handleStream = () => {
//     {
//       outputVideo && setVideoApi(`${API_URL}/output/${outputVideo}`);
//     }
//     console.log(videoApi);
//   };

//   // Handle successful login
//   const handleLoginSuccess = () => {
//     setIsAuthenticated(true);
//     setShowLogin(false);
//     if (videoFile) {
//       handleUpload();
//     }
//   };

//   return (
//     <section>
//       <Element name="create">
//         <div className="container">
//           <div className="flex flex-col items-center px-0 py-20 space-y-4 m-10 border-8 border-double border-[#34477C] rounded-2xl max-w-fit mx-auto ">
//             <div className="flex flex-col gap-4 items-center justify-center mx-[190px]">
//               <div className="w-auto">
//                 <h1 className="mb-2 h1 text-p4 uppercase max-lg:mb-2 max-lg:h2 max-md:mb-2 max-md:text-5xl max-md:leading-12">
//                   Make it happen!
//                 </h1>
//                 <p className="max-w-fit mb-6 body-1 max-md:mb-10 mx-auto">
//                   ⚽Get your own video and realize its analytics.⚽
//                 </p>
//                 <div className="w-512 mx-auto">
//                   <Dropzone onDrop={handleDrop} accept={{ "video/*": [] }}>
//                     {({ getRootProps, getInputProps }) => (
//                       <div
//                         {...getRootProps()}
//                         className="border-4 border-dashed border-blue-500 p-6 rounded-md text-center cursor-pointer hover:bg-blue-50"
//                       >
//                         <input {...getInputProps()} />
//                         <p className="text-gray-600">
//                           Drag & Drop a video file here, or click to select
//                         </p>
//                       </div>
//                     )}
//                   </Dropzone>
//                 </div>
//               </div>

//               <Button
//                 onClick={handleUploadClick}
//                 disabled={!videoFile}
//                 icon={buttonIcon.src}
//               >
//                 Upload and Process
//               </Button>

//               {/* Progress Bar */}
//               {progress > 0 && (
//                 <div className="w-[30%] text-center">
//                   <p className="text-gray-600 mb-1">
//                     {" "}
//                     {progress === 100
//                       ? "Video uploaded, wait for response..."
//                       : `Uploading ${progress}%...`}
//                   </p>
//                   {progress !== 100 && (
//                     <div className="w-full bg-gray-300 rounded-full h-4">
//                       <div
//                         className="bg-blue-500 h-4 rounded-full transition-all duration-200 ease-in-out"
//                         style={{ width: `${progress}%` }}
//                       ></div>
//                     </div>
//                   )}
//                 </div>
//               )}

//               {outputVideo && (
//                 <Dialog>
//                   <DialogTrigger asChild>
//                     <Button icon={videoViewIcon.src} onClick={handleStream}>
//                       View Processed video
//                     </Button>
//                   </DialogTrigger>

//                   <DialogContent className="shad-dialog g7">
//                     <DialogHeader>
//                       <DialogTitle>Football Analytics Video</DialogTitle>
//                       <DialogDescription>
//                         You can share this with anyone who has this link.
//                       </DialogDescription>
//                     </DialogHeader>
//                     <video
//                       name="outputVideo"
//                       width={855}
//                       height={655}
//                       className="rounded-xl"
//                       autoPlay
//                       muted
//                       loop
//                       controls
//                       onError={(e) => {
//                         alert("Failed to load the video.");
//                         console.log("Error e.target.error =", e.target.error);
//                         console.log(videoApi);
//                       }}
//                     >
//                       <source id="video-source" src={videoApi} type="video/mp4" />
//                     </video>

//                     <div className="flex items-center space-x-2">
//                       <div className="grid flex-1 gap-2">
//                         <Label htmlFor="link" className="sr-only">
//                           Link
//                         </Label>
//                         <Input id="link" defaultValue={videoApi} readOnly className="g7" />
//                       </div>
//                       <button
//                         type="submit"
//                         size="sm"
//                         className="px-3"
//                         aria-label="Copy link to clipboard"
//                         onClick={handleCopyLink}
//                       >
//                         <span className="sr-only">Copy</span>
//                         <Copy />
//                       </button>
//                     </div>
//                     <DialogFooter className="sm:justify-start">
//                       <button
//                         onClick={() => {
//                           if (outputVideo) {
//                             const filename = videoApi.split("/").pop() || "processed_test.mp4";
//                             const downloadLink = `${API_URL}/download/${filename}`;
//                             console.log(filename);
//                             console.log(downloadLink);

//                             window.open(downloadLink, "_blank").focus;
//                           } else {
//                             alert("No video available for download.");
//                           }
//                         }}
//                         className="px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
//                       >
//                         DOWNLOAD
//                       </button>
//                     </DialogFooter>
//                   </DialogContent>
//                 </Dialog>
//               )}
//               <p className="text-sm text-gray-500">
//                 Video processing may take up to 2 minutes. Feel free to leave
//                 this page and come back later!
//               </p>
//             </div>
//           </div>
//         </div>
//       </Element>

//       {/* Login Modal */}
//       <LoginModal 
//         isOpen={showLogin} 
//         onClose={() => setShowLogin(false)}
//         onLoginSuccess={handleLoginSuccess}
//       />
//     </section>
//   );
// };

// // export default VideoUploader;
// import { useState } from "react";
// import axios from "axios";
// import Dropzone from "react-dropzone";
// import "dropzone/dist/dropzone.css";
// import Button from "../components/Button";
// import { Element } from "react-scroll";
// import { API_URL } from "../api";
// import { Copy } from "lucide-react";
// import LoginModal from "../components/LoginModal";

// import {
//   Dialog,
//   DialogContent,
//   DialogDescription,
//   DialogFooter,
//   DialogHeader,
//   DialogTitle,
//   DialogTrigger,
// } from "@/components/ui/dialog";
// import { Input } from "@/components/ui/input";
// import { Label } from "@/components/ui/label";

// const VideoUploader = () => {
//   const [videoFile, setVideoFile] = useState(null);
//   const [outputVideo, setOutputVideo] = useState(null);
//   const [videoApi, setVideoApi] = useState(null);
//   const [progress, setProgress] = useState(0);
//   const [showLogin, setShowLogin] = useState(false);
//   const [isAuthenticated, setIsAuthenticated] = useState(
//     !!localStorage.getItem('token') // Check if token already exists
//   );

//   const buttonIcon = {
//     src: "/images/plan-1.png",
//     alt: "button Logo",
//   };

//   const videoViewIcon = {
//     src: "/images/plan-2.png",
//     alt: "view video logo",
//   };

//   const handleCopyLink = () => {
//     navigator.clipboard.writeText(videoApi);
//     alert("Link copied to clipboard!");
//   };

//   // Handle file drop
//   const handleDrop = (acceptedFiles) => {
//     if (acceptedFiles.length > 0) {
//       setVideoFile(acceptedFiles[0]);
//       console.log("File selected:", acceptedFiles[0].name);
//     } else {
//       alert("Please drop a valid video file (mp4, avi, mov, mkv)");
//     }
//   };

//   // Handle upload button click
//   const handleUploadClick = () => {
//     if (!videoFile) {
//       alert("Please select a video file first!");
//       return;
//     }
//     if (!isAuthenticated) {
//       setShowLogin(true);
//       return;
//     }
//     handleUpload();
//   };

//   // Handle file upload
//   const handleUpload = async () => {
//     if (!videoFile) return;

//     const token = localStorage.getItem('token');
//     if (!token) {
//       setShowLogin(true);
//       return;
//     }

//     const formData = new FormData();
//     formData.append("file", videoFile);

//     try {
//       const uploadResponse = await axios.post(
//         `${API_URL}/upload-video/`,
//         formData,
//         {
//           headers: {
//             "Content-Type": "multipart/form-data",
//             "Authorization": `Bearer ${token}` // Send token with upload
//           },
//           onUploadProgress: (event) => {
//             const percentCompleted = Math.round(
//               (event.loaded * 100) / event.total
//             );
//             setProgress(percentCompleted);
//           },
//         }
//       );
//       setOutputVideo(uploadResponse.data.video_url);
//       console.log("Video uploaded:", uploadResponse.data);
//     } catch (error) {
//       console.error("Error uploading video", error);
//       if (error.response && error.response.status === 401) {
//         alert("Session expired. Please login again.");
//         localStorage.removeItem('token');
//         setIsAuthenticated(false);
//         setShowLogin(true);
//       } else {
//         alert("Failed to upload the video. Please try again.");
//       }
//     } finally {
//       setProgress(0);
//     }
//   };

//   const handleStream = () => {
//     {
//       outputVideo && setVideoApi(`${API_URL}/output/${outputVideo}`);
//     }
//     console.log(videoApi);
//   };

//   // Handle successful login
//   const handleLoginSuccess = () => {
//     setIsAuthenticated(true);
//     setShowLogin(false);
//     // Auto upload after login if file exists
//     if (videoFile) {
//       setTimeout(() => handleUpload(), 500);
//     }
//   };

//   return (
//     <section>
//       <Element name="create">
//         <div className="container">
//           <div className="flex flex-col items-center px-0 py-20 space-y-4 m-10 border-8 border-double border-[#34477C] rounded-2xl max-w-fit mx-auto">
//             <div className="flex flex-col gap-4 items-center justify-center mx-[190px]">
//               <div className="w-auto">
//                 <h1 className="mb-2 h1 text-p4 uppercase max-lg:mb-2 max-lg:h2 max-md:mb-2 max-md:text-5xl max-md:leading-12">
//                   Make it happen!
//                 </h1>
//                 <p className="max-w-fit mb-6 body-1 max-md:mb-10 mx-auto">
//                   ⚽Get your own video and realize its analytics.⚽
//                 </p>
//                 <div className="w-512 mx-auto">
//                   <Dropzone onDrop={handleDrop} accept={{ "video/*": [] }}>
//                     {({ getRootProps, getInputProps, isDragActive }) => (
//                       <div
//                         {...getRootProps()}
//                         className={`border-4 border-dashed rounded-md text-center cursor-pointer p-6 transition-all
//                           ${isDragActive 
//                             ? "border-cyan-400 bg-cyan-50/10 scale-105" 
//                             : "border-blue-500 hover:bg-blue-50/10"
//                           }`}
//                       >
//                         <input {...getInputProps()} />
//                         <p className="text-gray-600">
//                           {videoFile
//                             ? `✅ Selected: ${videoFile.name}`
//                             : isDragActive
//                               ? "Drop the video here..."
//                               : "Drag & Drop a video file here, or click to select"
//                           }
//                         </p>
//                       </div>
//                     )}
//                   </Dropzone>
//                 </div>
//               </div>

//               <Button
//                 onClick={handleUploadClick}
//                 disabled={!videoFile}
//                 icon={buttonIcon.src}
//               >
//                 Upload and Process
//               </Button>

//               {/* Progress Bar */}
//               {progress > 0 && (
//                 <div className="w-[30%] text-center">
//                   <p className="text-gray-600 mb-1">
//                     {progress === 100
//                       ? "Video uploaded, wait for response..."
//                       : `Uploading ${progress}%...`}
//                   </p>
//                   {progress !== 100 && (
//                     <div className="w-full bg-gray-300 rounded-full h-4">
//                       <div
//                         className="bg-blue-500 h-4 rounded-full transition-all duration-200 ease-in-out"
//                         style={{ width: `${progress}%` }}
//                       ></div>
//                     </div>
//                   )}
//                 </div>
//               )}

//               {outputVideo && (
//                 <Dialog>
//                   <DialogTrigger asChild>
//                     <Button icon={videoViewIcon.src} onClick={handleStream}>
//                       View Processed video
//                     </Button>
//                   </DialogTrigger>

//                   <DialogContent className="shad-dialog g7">
//                     <DialogHeader>
//                       <DialogTitle>Football Analytics Video</DialogTitle>
//                       <DialogDescription>
//                         You can share this with anyone who has this link.
//                       </DialogDescription>
//                     </DialogHeader>
//                     <video
//                       name="outputVideo"
//                       width={855}
//                       height={655}
//                       className="rounded-xl"
//                       autoPlay
//                       muted
//                       loop
//                       controls
//                       onError={(e) => {
//                         alert("Failed to load the video.");
//                         console.log("Error e.target.error =", e.target.error);
//                       }}
//                     >
//                       <source id="video-source" src={videoApi} type="video/mp4" />
//                     </video>

//                     <div className="flex items-center space-x-2">
//                       <div className="grid flex-1 gap-2">
//                         <Label htmlFor="link" className="sr-only">
//                           Link
//                         </Label>
//                         <Input id="link" defaultValue={videoApi} readOnly className="g7" />
//                       </div>
//                       <button
//                         type="submit"
//                         size="sm"
//                         className="px-3"
//                         aria-label="Copy link to clipboard"
//                         onClick={handleCopyLink}
//                       >
//                         <span className="sr-only">Copy</span>
//                         <Copy />
//                       </button>
//                     </div>
//                     <DialogFooter className="sm:justify-start">
//                       <button
//                         onClick={() => {
//                           if (outputVideo) {
//                             const filename = videoApi.split("/").pop() || "processed_test.mp4";
//                             const downloadLink = `${API_URL}/download/${filename}`;
//                             window.open(downloadLink, "_blank").focus;
//                           } else {
//                             alert("No video available for download.");
//                           }
//                         }}
//                         className="px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
//                       >
//                         DOWNLOAD
//                       </button>
//                     </DialogFooter>
//                   </DialogContent>
//                 </Dialog>
//               )}

//               <p className="text-sm text-gray-500">
//                 Video processing may take up to 2 minutes. Feel free to leave
//                 this page and come back later!
//               </p>
//             </div>
//           </div>
//         </div>
//       </Element>

//       {/* Login Modal */}
//       <LoginModal
//         isOpen={showLogin}
//         onClose={() => setShowLogin(false)}
//         onLoginSuccess={handleLoginSuccess}
//       />
//     </section>
//   );
// };

// export default VideoUploader;
// import { useState } from "react";
// import axios from "axios";
// import Dropzone from "react-dropzone";
// import "dropzone/dist/dropzone.css";
// import Button from "../components/Button";
// import { Element } from "react-scroll";
// import { API_URL } from "../api";
// import { Copy } from "lucide-react";
// import LoginModal from "../components/LoginModal";

// import {
//   Dialog,
//   DialogContent,
//   DialogDescription,
//   DialogFooter,
//   DialogHeader,
//   DialogTitle,
//   DialogTrigger,
// } from "@/components/ui/dialog";
// import { Input } from "@/components/ui/input";
// import { Label } from "@/components/ui/label";

// const VideoUploader = () => {
//   const [videoFile, setVideoFile] = useState(null);
//   const [outputVideo, setOutputVideo] = useState(null);
//   const [videoApi, setVideoApi] = useState(null);
//   const [progress, setProgress] = useState(0);
//   const [showLogin, setShowLogin] = useState(false);
//   const [isAuthenticated, setIsAuthenticated] = useState(
//     !!localStorage.getItem('token')
//   );

//   const buttonIcon = {
//     src: "/images/plan-1.png",
//     alt: "button Logo",
//   };

//   const videoViewIcon = {
//     src: "/images/plan-2.png",
//     alt: "view video logo",
//   };

//   const handleCopyLink = () => {
//     navigator.clipboard.writeText(videoApi);
//     alert("Link copied to clipboard!");
//   };

//   // Handle file drop
//   const handleDrop = (acceptedFiles) => {
//     if (acceptedFiles.length > 0) {
//       setVideoFile(acceptedFiles[0]);
//       console.log("File selected:", acceptedFiles[0].name);
//     } else {
//       alert("Please drop a valid video file (mp4, avi, mov, mkv)");
//     }
//   };

//   // Handle upload button click
//   const handleUploadClick = () => {
//     if (!videoFile) {
//       alert("Please select a video file first!");
//       return;
//     }
//     if (!isAuthenticated) {
//       setShowLogin(true);
//       return;
//     }
//     handleUpload();
//   };

//   // Handle file upload
//   const handleUpload = async () => {
//     if (!videoFile) return;

//     const token = localStorage.getItem('token');
//     if (!token) {
//       setShowLogin(true);
//       return;
//     }

//     const formData = new FormData();
//     formData.append("file", videoFile);

//     try {
//       const uploadResponse = await axios.post(
//         `${API_URL}/upload-video/`,
//         formData,
//         {
//           headers: {
//             "Content-Type": "multipart/form-data",
//             "Authorization": `Bearer ${token}`
//           },
//           onUploadProgress: (event) => {
//             const percentCompleted = Math.round(
//               (event.loaded * 100) / event.total
//             );
//             setProgress(percentCompleted);
//           },
//         }
//       );
//       setOutputVideo(uploadResponse.data.video_url);
//       console.log("Video uploaded:", uploadResponse.data);
//     } catch (error) {
//       console.error("Error uploading video", error);
//       if (error.response && error.response.status === 401) {
//         alert("Session expired. Please login again.");
//         localStorage.removeItem('token');
//         setIsAuthenticated(false);
//         setShowLogin(true);
//       } else {
//         alert("Failed to upload the video. Please try again.");
//       }
//     } finally {
//       setProgress(0);
//     }
//   };

//   const handleStream = () => {
//     outputVideo && setVideoApi(`${API_URL}/output/${outputVideo}`);
//     console.log(videoApi);
//   };

//   // Handle successful login
//   const handleLoginSuccess = () => {
//     setIsAuthenticated(true);
//     setShowLogin(false);
//     if (videoFile) {
//       setTimeout(() => handleUpload(), 500);
//     }
//   };

//   return (
//     <section>
//       <Element name="create">
//         <div className="container">
//           <div className="flex flex-col items-center px-0 py-20 space-y-4 m-10 border-8 border-double border-[#34477C] rounded-2xl max-w-fit mx-auto">
//             <div className="flex flex-col gap-4 items-center justify-center mx-[190px]">
//               <div className="w-auto">
//                 <h1 className="mb-2 h1 text-p4 uppercase max-lg:mb-2 max-lg:h2 max-md:mb-2 max-md:text-5xl max-md:leading-12">
//                   Make it happen!
//                 </h1>
//                 <p className="max-w-fit mb-6 body-1 max-md:mb-10 mx-auto">
//                   ⚽Get your own video and realize its analytics.⚽
//                 </p>
//                 <div className="w-512 mx-auto">
//                   <Dropzone onDrop={handleDrop} accept={{ "video/*": [] }}>
//                     {({ getRootProps, getInputProps, isDragActive }) => (
//                       <div
//                         {...getRootProps()}
//                         className={`border-4 border-dashed rounded-md text-center cursor-pointer p-6 transition-all
//                           ${isDragActive
//                             ? "border-cyan-400 bg-cyan-50/10 scale-105"
//                             : "border-blue-500 hover:bg-blue-50/10"
//                           }`}
//                       >
//                         <input {...getInputProps()} />
//                         <p className="text-gray-600">
//                           {videoFile
//                             ? `✅ Selected: ${videoFile.name}`
//                             : isDragActive
//                               ? "Drop the video here..."
//                               : "Drag & Drop a video file here, or click to select"
//                           }
//                         </p>
//                       </div>
//                     )}
//                   </Dropzone>
//                 </div>
//               </div>

//               <Button
//                 onClick={handleUploadClick}
//                 disabled={!videoFile}
//                 icon={buttonIcon.src}
//               >
//                 Upload and Process
//               </Button>

//               {/* Progress Bar */}
//               {progress > 0 && (
//                 <div className="w-[30%] text-center">
//                   <p className="text-gray-600 mb-1">
//                     {progress === 100
//                       ? "Video uploaded, wait for response..."
//                       : `Uploading ${progress}%...`}
//                   </p>
//                   {progress !== 100 && (
//                     <div className="w-full bg-gray-300 rounded-full h-4">
//                       <div
//                         className="bg-blue-500 h-4 rounded-full transition-all duration-200 ease-in-out"
//                         style={{ width: `${progress}%` }}
//                       ></div>
//                     </div>
//                   )}
//                 </div>
//               )}

//               {outputVideo && (
//                 <Dialog>
//                   <DialogTrigger asChild>
//                     <Button icon={videoViewIcon.src} onClick={handleStream}>
//                       View Processed video
//                     </Button>
//                   </DialogTrigger>

//                   <DialogContent className="shad-dialog g7">
//                     <DialogHeader>
//                       <DialogTitle>Football Analytics Video</DialogTitle>
//                       <DialogDescription>
//                         You can share this with anyone who has this link.
//                       </DialogDescription>
//                     </DialogHeader>
//                     <video
//                       name="outputVideo"
//                       width={855}
//                       height={655}
//                       className="rounded-xl"
//                       autoPlay
//                       muted
//                       loop
//                       controls
//                       onError={(e) => {
//                         alert("Failed to load the video.");
//                         console.log("Error:", e.target.error);
//                       }}
//                     >
//                       <source id="video-source" src={videoApi} type="video/mp4" />
//                     </video>

//                     <div className="flex items-center space-x-2">
//                       <div className="grid flex-1 gap-2">
//                         <Label htmlFor="link" className="sr-only">Link</Label>
//                         <Input id="link" defaultValue={videoApi} readOnly className="g7" />
//                       </div>
//                       <button
//                         size="sm"
//                         className="px-3"
//                         aria-label="Copy link to clipboard"
//                         onClick={handleCopyLink}
//                       >
//                         <span className="sr-only">Copy</span>
//                         <Copy />
//                       </button>
//                     </div>
//                     <DialogFooter className="sm:justify-start">
//                       <button
//                         onClick={() => {
//                           if (outputVideo) {
//                             const filename = videoApi.split("/").pop() || "processed_test.mp4";
//                             const downloadLink = `${API_URL}/download/${filename}`;
//                             window.open(downloadLink, "_blank").focus;
//                           } else {
//                             alert("No video available for download.");
//                           }
//                         }}
//                         className="px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
//                       >
//                         DOWNLOAD
//                       </button>
//                     </DialogFooter>
//                   </DialogContent>
//                 </Dialog>
//               )}

//               <p className="text-sm text-gray-500">
//                 Video processing may take up to 2 minutes. Feel free to leave
//                 this page and come back later!
//               </p>
//             </div>
//           </div>
//         </div>
//       </Element>

//       {/* Login Modal */}
//       <LoginModal
//         isOpen={showLogin}
//         onClose={() => setShowLogin(false)}
//         onLoginSuccess={handleLoginSuccess}
//       />
//     </section>
//   );
// };

// export default VideoUploader;


// import { useState, useEffect, useRef } from "react";
// import axios from "axios";
// import Dropzone from "react-dropzone";
// import "dropzone/dist/dropzone.css";
// import Button from "../components/Button";
// import { Element } from "react-scroll";
// import { API_URL } from "../api";
// import { Copy } from "lucide-react";
// import { useGoogleLogin } from "@react-oauth/google";

// import {
//   Dialog,
//   DialogContent,
//   DialogDescription,
//   DialogFooter,
//   DialogHeader,
//   DialogTitle,
//   DialogTrigger,
// } from "@/components/ui/dialog";
// import { Input } from "@/components/ui/input";
// import { Label } from "@/components/ui/label";

// import LoginModal from "../components/LoginModal.jsx";

// const VideoUploader = () => {
//   const [videoFile, setVideoFile] = useState(null);
//   const [outputVideo, setOutputVideo] = useState(null);
//   const [videoApi, setVideoApi] = useState(null);
//   const [progress, setProgress] = useState(0);
//   const [isAuthenticated, setIsAuthenticated] = useState(false);
//   const [showLoginModal, setShowLoginModal] = useState(false);

//   // Go Live state
//   const [showLiveModal, setShowLiveModal] = useState(false);
//   const [ipUrl, setIpUrl] = useState("http://192.168.1.5:8080/video");
//   const [isStreaming, setIsStreaming] = useState(false);
//   const [liveLoading, setLiveLoading] = useState(false);
//   const [liveError, setLiveError] = useState("");
//   const [trackingData, setTrackingData] = useState(null);
//   const dataInterval = useRef(null);

//   const buttonIcon = { src: "/images/plan-1.png", alt: "button Logo" };
//   const videoViewIcon = { src: "/images/plan-2.png", alt: "view video logo" };

//   // Poll /live/data every second while streaming
//   useEffect(() => {
//     if (isStreaming) {
//       dataInterval.current = setInterval(async () => {
//         try {
//           const res = await axios.get(`${API_URL}/live/data`);
//           setTrackingData(res.data);
//         } catch {}
//       }, 1000);
//     } else {
//       clearInterval(dataInterval.current);
//       setTrackingData(null);
//     }
//     return () => clearInterval(dataInterval.current);
//   }, [isStreaming]);

//   const handleCopyLink = () => {
//     navigator.clipboard.writeText(videoApi);
//     alert("Link copied to clipboard!");
//   };

//   const handleDrop = (acceptedFiles) => {
//     if (acceptedFiles.length > 0) {
//       setVideoFile(acceptedFiles[0]);
//     } else {
//       alert("Please drop a valid video file (mp4, avi, mov, mkv)");
//     }
//   };

//   const handleUpload = async () => {
//     if (!videoFile) return;
//     const formData = new FormData();
//     formData.append("file", videoFile);
//     try {
//       const uploadResponse = await axios.post(`${API_URL}/upload-video/`, formData, {
//         headers: { "Content-Type": "multipart/form-data" },
//         onUploadProgress: (event) => {
//           const percentCompleted = Math.round((event.loaded * 100) / event.total);
//           setProgress(percentCompleted);
//         },
//         timeout: 300000,
//       });
//       const videoUrl = uploadResponse.data.video_url;
//       const filename = videoUrl.split("/").pop();
//       setOutputVideo(filename);
//       alert("Video uploaded and processing started!");
//     } catch (error) {
//       console.error("Error uploading video", error);
//       alert("Failed to upload video. See console for details.");
//     } finally {
//       setProgress(0);
//     }
//   };

//   const handleStream = () => {
//     if (outputVideo) {
//       setVideoApi(`${API_URL}/output/${outputVideo}`);
//     }
//   };

//   const login = useGoogleLogin({
//     onSuccess: (credentialResponse) => {
//       setIsAuthenticated(true);
//       setShowLoginModal(false);
//       handleUpload();
//     },
//     onError: () => {
//       alert("Google login failed. Please try again.");
//     },
//   });

//   const handleUploadClick = () => {
//     if (!videoFile) {
//       alert("Please select a video file first!");
//       return;
//     }
//     if (!isAuthenticated) {
//       setShowLoginModal(true);
//     } else {
//       handleUpload();
//     }
//   };

//   // Live handlers
//   const handleGoLive = () => {
//     setLiveError("");
//     setShowLiveModal(true);
//   };

//   const startLive = async () => {
//     if (!ipUrl.trim()) {
//       setLiveError("Please enter your IPWebcam URL.");
//       return;
//     }
//     setLiveLoading(true);
//     setLiveError("");
//     try {
//       await axios.post(`${API_URL}/live/start?ip_url=${encodeURIComponent(ipUrl)}`);
//       setIsStreaming(true);
//     } catch {
//       setLiveError("Cannot connect. Check IP address and make sure phone & PC are on the same WiFi.");
//     } finally {
//       setLiveLoading(false);
//     }
//   };

//   const stopLive = async () => {
//     try {
//       await axios.post(`${API_URL}/live/stop`);
//     } catch {}
//     setIsStreaming(false);
//     setShowLiveModal(false);
//   };

//   return (
//     <section>
//       <Element name="create">
//         <div className="container">
//           <div className="flex flex-col items-center px-0 py-20 space-y-4 m-10 border-8 border-double border-[#34477C] rounded-2xl max-w-fit mx-auto">
//             <div className="flex flex-col gap-4 items-center justify-center mx-[190px]">
//               <h1 className="mb-2 h1 text-p4 uppercase max-lg:mb-2 max-lg:h2 max-md:mb-2 max-md:text-5xl max-md:leading-12">
//                 Make it happen!
//               </h1>
//               <p className="max-w-fit mb-6 body-1 max-md:mb-10 mx-auto">
//                 ⚽ Get your own video and realize its analytics. ⚽
//               </p>

//               {/* Dropzone */}
//               <div className="w-512 mx-auto">
//                 <Dropzone onDrop={handleDrop} accept={{ "video/*": [] }}>
//                   {({ getRootProps, getInputProps, isDragActive }) => (
//                     <div
//                       {...getRootProps()}
//                       className={`border-4 border-dashed rounded-md text-center cursor-pointer p-6 transition-all
//                         ${isDragActive ? "border-cyan-400 bg-cyan-50/10 scale-105" : "border-blue-500 hover:bg-blue-50/10"}`}
//                     >
//                       <input {...getInputProps()} />
//                       <p className="text-gray-600">
//                         {videoFile
//                           ? `✅ Selected: ${videoFile.name}`
//                           : isDragActive
//                           ? "Drop the video here..."
//                           : "Drag & Drop a video file here, or click to select"}
//                       </p>
//                     </div>
//                   )}
//                 </Dropzone>
//               </div>

//               {/* Upload Button */}
//               <Button
//                 onClick={handleUploadClick}
//                 disabled={!videoFile || progress > 0}
//                 icon={buttonIcon.src}
//               >
//                 {progress > 0 ? "Processing..." : "Upload and Process"}
//               </Button>

//               {/* Progress Bar */}
//               {progress > 0 && (
//                 <div className="w-[30%] text-center">
//                   <p className="text-gray-600 mb-1">
//                     {progress === 100
//                       ? "Video uploaded, processing... This may take 2+ minutes."
//                       : `Uploading ${progress}%...`}
//                   </p>
//                   {progress !== 100 && (
//                     <div className="w-full bg-gray-300 rounded-full h-4">
//                       <div
//                         className="bg-blue-500 h-4 rounded-full transition-all duration-200 ease-in-out"
//                         style={{ width: `${progress}%` }}
//                       ></div>
//                     </div>
//                   )}
//                 </div>
//               )}

//               {/* Output Video */}
//               {outputVideo && (
//                 <Dialog>
//                   <DialogTrigger asChild>
//                     <Button icon={videoViewIcon.src} onClick={handleStream}>
//                       View Processed Video
//                     </Button>
//                   </DialogTrigger>
//                   <DialogContent className="shad-dialog g7">
//                     <DialogHeader>
//                       <DialogTitle>Football Analytics Video</DialogTitle>
//                       <DialogDescription>
//                         You can share this with anyone who has this link.
//                       </DialogDescription>
//                     </DialogHeader>
//                     <video
//                       name="outputVideo"
//                       width={855}
//                       height={655}
//                       className="rounded-xl"
//                       autoPlay
//                       muted
//                       loop
//                       controls
//                       onError={(e) => {
//                         alert("Failed to load the video.");
//                         console.log("Video Error:", e.target.error);
//                       }}
//                     >
//                       <source id="video-source" src={videoApi} type="video/mp4" />
//                     </video>
//                     <div className="flex items-center space-x-2">
//                       <div className="grid flex-1 gap-2">
//                         <Label htmlFor="link" className="sr-only">Link</Label>
//                         <Input id="link" defaultValue={videoApi} readOnly className="g7" />
//                       </div>
//                       <button size="sm" className="px-3" aria-label="Copy link" onClick={handleCopyLink}>
//                         <Copy />
//                       </button>
//                     </div>
//                     <DialogFooter className="sm:justify-start">
//                       <button
//                         onClick={() => {
//                           if (outputVideo) {
//                             window.open(`${API_URL}/download/${outputVideo}`, "_blank");
//                           } else {
//                             alert("No video available for download.");
//                           }
//                         }}
//                         className="px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
//                       >
//                         DOWNLOAD
//                       </button>
//                     </DialogFooter>
//                   </DialogContent>
//                 </Dialog>
//               )}

//               {/* ── GO LIVE Button ────────────────────────────────────── */}
//               <button
//                 onClick={handleGoLive}
//                 style={{
//                   display: "flex", alignItems: "center", justifyContent: "center",
//                   gap: "12px",
//                   background: "linear-gradient(135deg, #0f3460, #1a1a4e)",
//                   border: "2px solid #e63946",
//                   color: "#ffffff",
//                   fontSize: "15px", fontWeight: "800", letterSpacing: "2.5px",
//                   padding: "15px 40px", borderRadius: "50px",
//                   cursor: "pointer", width: "100%", maxWidth: "420px",
//                   transition: "all 0.3s ease",
//                 }}
//                 onMouseEnter={e => {
//                   e.currentTarget.style.background = "linear-gradient(135deg, #e63946, #c1121f)";
//                   e.currentTarget.style.boxShadow = "0 0 24px rgba(230,57,70,0.55)";
//                   e.currentTarget.style.transform = "translateY(-2px)";
//                 }}
//                 onMouseLeave={e => {
//                   e.currentTarget.style.background = "linear-gradient(135deg, #0f3460, #1a1a4e)";
//                   e.currentTarget.style.boxShadow = "none";
//                   e.currentTarget.style.transform = "translateY(0)";
//                 }}
//               >
//                 <span style={{
//                   width: "34px", height: "34px",
//                   background: "rgba(230,57,70,0.2)", borderRadius: "50%",
//                   display: "flex", alignItems: "center", justifyContent: "center",
//                 }}>
//                   <span style={{
//                     width: "13px", height: "13px",
//                     background: "#e63946", borderRadius: "50%",
//                     animation: "livePulse 1.2s infinite",
//                   }} />
//                 </span>
//                 GO LIVE
//               </button>

//               <p className="text-sm text-gray-500">
//                 Video processing may take up to 2 minutes. Feel free to leave
//                 this page and come back later!
//               </p>
//             </div>
//           </div>
//         </div>
//       </Element>

//       {/* Login Modal */}
//       <LoginModal
//         isOpen={showLoginModal}
//         onClose={() => setShowLoginModal(false)}
//         onGoogleLoginSuccess={() => {
//           setIsAuthenticated(true);
//           handleUpload();
//         }}
//       />

//       {/* ── Live Tracking Modal ─────────────────────────────────────────── */}
//       {showLiveModal && (
//         <div
//           onClick={() => { if (!isStreaming) setShowLiveModal(false); }}
//           style={{
//             position: "fixed", inset: 0,
//             background: "rgba(0,0,0,0.88)",
//             display: "flex", alignItems: "center", justifyContent: "center",
//             zIndex: 9999, padding: "16px",
//           }}
//         >
//           <div
//             onClick={e => e.stopPropagation()}
//             style={{
//               background: "#0d1b2a", border: "1px solid #1e3a5f",
//               borderRadius: "16px", padding: "28px",
//               width: "100%", maxWidth: "820px",
//               maxHeight: "90vh", overflowY: "auto",
//               boxShadow: "0 0 50px rgba(0,150,255,0.12)",
//             }}
//           >
//             {/* Modal Header */}
//             <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "22px" }}>
//               <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
//                 {isStreaming && (
//                   <span style={{ color: "#e63946", fontWeight: 800, fontSize: "12px", letterSpacing: "2px", animation: "blink 1s infinite" }}>
//                     ● LIVE
//                   </span>
//                 )}
//                 <h2 style={{ color: "#fff", fontSize: "19px", fontWeight: 700, margin: 0 }}>
//                   Live Player &amp; Ball Tracking
//                 </h2>
//               </div>
//               {!isStreaming && (
//                 <button
//                   onClick={() => setShowLiveModal(false)}
//                   style={{
//                     background: "none", border: "1px solid #2a3a4a",
//                     color: "#888", width: "32px", height: "32px",
//                     borderRadius: "50%", cursor: "pointer", fontSize: "13px",
//                   }}
//                 >✕</button>
//               )}
//             </div>

//             {/* Before streaming */}
//             {!isStreaming && (
//               <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
//                 <label style={{ color: "#8899aa", fontSize: "13px", fontWeight: 600 }}>
//                   IPWebcam Stream URL
//                 </label>
//                 <input
//                   type="text"
//                   value={ipUrl}
//                   onChange={e => setIpUrl(e.target.value)}
//                   placeholder="http://192.168.x.x:8080/video"
//                   style={{
//                     width: "100%", padding: "12px 16px",
//                     background: "#132337", border: "1px solid #1e3a5f",
//                     borderRadius: "8px", color: "#fff",
//                     fontSize: "14px", boxSizing: "border-box",
//                   }}
//                 />
//                 {liveError && (
//                   <p style={{ color: "#e63946", fontSize: "13px", margin: 0 }}>{liveError}</p>
//                 )}
//                 <button
//                   onClick={startLive}
//                   disabled={liveLoading}
//                   style={{
//                     width: "100%", padding: "14px",
//                     background: "linear-gradient(135deg, #0096ff, #0052cc)",
//                     color: "#fff", fontWeight: 800, fontSize: "14px",
//                     letterSpacing: "2px", border: "none", borderRadius: "8px",
//                     cursor: liveLoading ? "not-allowed" : "pointer",
//                     opacity: liveLoading ? 0.6 : 1,
//                   }}
//                 >
//                   {liveLoading ? "Connecting..." : "▶  START LIVE TRACKING"}
//                 </button>

//                 {/* Instructions */}
//                 <div style={{ background: "#0a141e", border: "1px solid #1e3a5f", borderRadius: "8px", padding: "14px 16px" }}>
//                   <p style={{ color: "#8899aa", fontSize: "13px", fontWeight: 600, margin: "0 0 8px 0" }}>📱 How to set up:</p>
//                   <ol style={{ color: "#6677aa", fontSize: "13px", paddingLeft: "18px", margin: 0, lineHeight: 2 }}>
//                     <li>Install <strong style={{ color: "#aabbcc" }}>IP Webcam</strong> app on your Android phone</li>
//                     <li>Connect phone &amp; PC to the <strong style={{ color: "#aabbcc" }}>same WiFi</strong></li>
//                     <li>Open app → scroll down → tap <strong style={{ color: "#aabbcc" }}>Start Server</strong></li>
//                     <li>Copy the IP shown e.g. <code style={{ background: "#1a2a3a", padding: "1px 5px", borderRadius: "3px", color: "#66aaff", fontSize: "12px" }}>http://192.168.1.5:8080</code></li>
//                     <li>Paste above with <code style={{ background: "#1a2a3a", padding: "1px 5px", borderRadius: "3px", color: "#66aaff", fontSize: "12px" }}>/video</code> at the end → click Start</li>
//                   </ol>
//                 </div>
//               </div>
//             )}

//             {/* While streaming */}
//             {isStreaming && (
//               <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>

//                 {/* Live feed */}
//                 <div style={{ borderRadius: "10px", overflow: "hidden", border: "2px solid #e63946" }}>
//                   <img
//                     src={`${API_URL}/live/feed`}
//                     alt="Live tracking feed"
//                     style={{ width: "100%", display: "block" }}
//                   />
//                 </div>

//                 {/* Stats */}
//                 {trackingData && (
//                   <div style={{ background: "#0a141e", border: "1px solid #1e3a5f", borderRadius: "10px", padding: "16px" }}>
//                     <p style={{ color: "#facc15", fontWeight: 700, fontSize: "14px", margin: "0 0 12px 0" }}>Tracking Stats</p>

//                     <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
//                       <span style={{ color: "#8899aa", fontSize: "13px" }}>👤 Players detected</span>
//                       <span style={{ color: "#22c55e", fontWeight: 700 }}>{trackingData.player_count ?? 0}</span>
//                     </div>

//                     <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
//                       <span style={{ color: "#8899aa", fontSize: "13px" }}>⚽ Ball</span>
//                       <span style={{ color: trackingData.ball_detected ? "#facc15" : "#4b5563", fontWeight: 700 }}>
//                         {trackingData.ball_detected ? "Detected" : "Not found"}
//                       </span>
//                     </div>

//                     {trackingData.ball_detected && trackingData.ball_position && (
//                       <p style={{ color: "#4b6080", fontSize: "11px", margin: "0 0 8px 0" }}>
//                         Position: ({trackingData.ball_position[0]}, {trackingData.ball_position[1]}) | Conf: {trackingData.ball?.confidence}
//                       </p>
//                     )}

//                     {trackingData.referees?.length > 0 && (
//                       <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
//                         <span style={{ color: "#8899aa", fontSize: "13px" }}>🟠 Referee</span>
//                         <span style={{ color: "#fb923c", fontWeight: 700 }}>{trackingData.referees.length}</span>
//                       </div>
//                     )}

//                     <hr style={{ border: "none", borderTop: "1px solid #1e3a5f", margin: "10px 0" }} />

//                     {trackingData.players?.length > 0 && (
//                       <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
//                         {trackingData.players.map((pl, i) => (
//                           <div key={i} style={{ background: "#132337", color: "#8899aa", fontSize: "11px", padding: "4px 8px", borderRadius: "4px" }}>
//                             Player {i + 1} — pos: ({pl.center[0]}, {pl.center[1]}) — conf: {pl.confidence}
//                           </div>
//                         ))}
//                       </div>
//                     )}

//                     <p style={{ color: "#2a3a4a", fontSize: "11px", margin: "8px 0 0 0" }}>
//                       Total objects: {trackingData.total_objects}
//                     </p>
//                   </div>
//                 )}

//                 {/* Stop button */}
//                 <button
//                   onClick={stopLive}
//                   style={{
//                     width: "100%", padding: "13px",
//                     background: "#e63946", color: "#fff",
//                     fontWeight: 800, fontSize: "14px", letterSpacing: "2px",
//                     border: "none", borderRadius: "8px", cursor: "pointer",
//                   }}
//                   onMouseEnter={e => e.currentTarget.style.background = "#c1121f"}
//                   onMouseLeave={e => e.currentTarget.style.background = "#e63946"}
//                 >
//                   ⏹  STOP LIVE
//                 </button>
//               </div>
//             )}
//           </div>
//         </div>
//       )}

//       {/* Animations */}
//       <style>{`
//         @keyframes livePulse {
//           0%   { box-shadow: 0 0 0 0 rgba(230,57,70,0.7); }
//           70%  { box-shadow: 0 0 0 10px rgba(230,57,70,0); }
//           100% { box-shadow: 0 0 0 0 rgba(230,57,70,0); }
//         }
//         @keyframes blink {
//           0%, 100% { opacity: 1; }
//           50%       { opacity: 0.2; }
//         }
//       `}</style>
//     </section>
//   );
// };

// export default VideoUploader;

// import { useState, useEffect, useRef } from "react";
// import axios from "axios";
// import Dropzone from "react-dropzone";
// import "dropzone/dist/dropzone.css";
// import Button from "../components/Button";
// import { Element } from "react-scroll";
// import { API_URL } from "../api";
// import { Copy } from "lucide-react";
// import { useGoogleLogin } from "@react-oauth/google";

// import {
//   Dialog,
//   DialogContent,
//   DialogDescription,
//   DialogFooter,
//   DialogHeader,
//   DialogTitle,
//   DialogTrigger,
// } from "@/components/ui/dialog";
// import { Input } from "@/components/ui/input";
// import { Label } from "@/components/ui/label";

// import LoginModal from "../components/LoginModal.jsx";

// const VideoUploader = () => {
//   const [videoFile, setVideoFile] = useState(null);
//   const [outputVideo, setOutputVideo] = useState(null);
//   const [videoApi, setVideoApi] = useState(null);
//   const [progress, setProgress] = useState(0);
//   const [isAuthenticated, setIsAuthenticated] = useState(false);
//   const [showLoginModal, setShowLoginModal] = useState(false);

//   // Go Live state
//   const [showLiveModal, setShowLiveModal] = useState(false);
//   const [ipUrl, setIpUrl] = useState("http://192.168.1.5:8080/video");
//   const [isStreaming, setIsStreaming] = useState(false);
//   const [liveLoading, setLiveLoading] = useState(false);
//   const [liveError, setLiveError] = useState("");
//   const [trackingData, setTrackingData] = useState(null);
//   const dataInterval = useRef(null);

//   const buttonIcon = { src: "/images/plan-1.png", alt: "button Logo" };
//   const videoViewIcon = { src: "/images/plan-2.png", alt: "view video logo" };

//   // Poll /live/data every second while streaming
//   useEffect(() => {
//     if (isStreaming) {
//       dataInterval.current = setInterval(async () => {
//         try {
//           const res = await axios.get(`${API_URL}/live/data`);
//           setTrackingData(res.data);
//         } catch {}
//       }, 1000);
//     } else {
//       clearInterval(dataInterval.current);
//       setTrackingData(null);
//     }
//     return () => clearInterval(dataInterval.current);
//   }, [isStreaming]);

//   const handleCopyLink = () => {
//     navigator.clipboard.writeText(videoApi);
//     alert("Link copied to clipboard!");
//   };

//   const handleDrop = (acceptedFiles) => {
//     if (acceptedFiles.length > 0) {
//       setVideoFile(acceptedFiles[0]);
//     } else {
//       alert("Please drop a valid video file (mp4, avi, mov, mkv)");
//     }
//   };

//   const handleUpload = async () => {
//     if (!videoFile) return;
//     const formData = new FormData();
//     formData.append("file", videoFile);
//     try {
//       const uploadResponse = await axios.post(`${API_URL}/upload-video/`, formData, {
//         headers: { "Content-Type": "multipart/form-data" },
//         onUploadProgress: (event) => {
//           const percentCompleted = Math.round((event.loaded * 100) / event.total);
//           setProgress(percentCompleted);
//         },
//         timeout: 300000,
//       });
//       const videoUrl = uploadResponse.data.video_url;
//       const filename = videoUrl.split("/").pop();
//       setOutputVideo(filename);
//       alert("Video uploaded and processing started!");
//     } catch (error) {
//       console.error("Error uploading video", error);
//       alert("Failed to upload video. See console for details.");
//     } finally {
//       setProgress(0);
//     }
//   };

//   const handleStream = () => {
//     if (outputVideo) {
//       setVideoApi(`${API_URL}/output/${outputVideo}`);
//     }
//   };

//   const login = useGoogleLogin({
//     onSuccess: (credentialResponse) => {
//       setIsAuthenticated(true);
//       setShowLoginModal(false);
//       handleUpload();
//     },
//     onError: () => {
//       alert("Google login failed. Please try again.");
//     },
//   });

//   const handleUploadClick = () => {
//     if (!videoFile) {
//       alert("Please select a video file first!");
//       return;
//     }
//     if (!isAuthenticated) {
//       setShowLoginModal(true);
//     } else {
//       handleUpload();
//     }
//   };

//   // Live handlers
//   const handleGoLive = () => {
//     setLiveError("");
//     setShowLiveModal(true);
//   };

//   const startLive = async () => {
//     if (!ipUrl.trim()) {
//       setLiveError("Please enter your IPWebcam URL.");
//       return;
//     }
//     setLiveLoading(true);
//     setLiveError("");
//     try {
//       await axios.post(`${API_URL}/live/start?ip_url=${encodeURIComponent(ipUrl)}`);
//       setIsStreaming(true);
//     } catch {
//       setLiveError("Cannot connect. Check IP address and make sure phone & PC are on the same WiFi.");
//     } finally {
//       setLiveLoading(false);
//     }
//   };

//   const stopLive = async () => {
//     try {
//       await axios.post(`${API_URL}/live/stop`);
//     } catch {}
//     setIsStreaming(false);
//     setShowLiveModal(false);
//   };

//   return (
//     <section>
//       <Element name="create">
//         <div className="container">
//           <div className="flex flex-col items-center px-0 py-20 space-y-4 m-10 border-8 border-double border-[#34477C] rounded-2xl max-w-fit mx-auto">
//             <div className="flex flex-col gap-4 items-center justify-center mx-[190px]">
//               <h1 className="mb-2 h1 text-p4 uppercase max-lg:mb-2 max-lg:h2 max-md:mb-2 max-md:text-5xl max-md:leading-12">
//                 Make it happen!
//               </h1>
//               <p className="max-w-fit mb-6 body-1 max-md:mb-10 mx-auto">
//                 ⚽ Get your own video and realize its analytics. ⚽
//               </p>

//               {/* Dropzone */}
//               <div className="w-512 mx-auto">
//                 <Dropzone onDrop={handleDrop} accept={{ "video/*": [] }}>
//                   {({ getRootProps, getInputProps, isDragActive }) => (
//                     <div
//                       {...getRootProps()}
//                       className={`border-4 border-dashed rounded-md text-center cursor-pointer p-6 transition-all
//                         ${isDragActive ? "border-cyan-400 bg-cyan-50/10 scale-105" : "border-blue-500 hover:bg-blue-50/10"}`}
//                     >
//                       <input {...getInputProps()} />
//                       <p className="text-gray-600">
//                         {videoFile
//                           ? `✅ Selected: ${videoFile.name}`
//                           : isDragActive
//                           ? "Drop the video here..."
//                           : "Drag & Drop a video file here, or click to select"}
//                       </p>
//                     </div>
//                   )}
//                 </Dropzone>
//               </div>

//               {/* Upload Button */}
//               <Button
//                 onClick={handleUploadClick}
//                 disabled={!videoFile || progress > 0}
//                 icon={buttonIcon.src}
//               >
//                 {progress > 0 ? "Processing..." : "Upload and Process"}
//               </Button>

//               {/* ── GO LIVE Button — always visible below Upload ── */}
//               <button
//                 onClick={handleGoLive}
//                 style={{
//                   display: "flex", alignItems: "center", justifyContent: "center",
//                   gap: "12px",
//                   background: "linear-gradient(135deg, #0f3460, #1a1a4e)",
//                   border: "2px solid #e63946",
//                   color: "#ffffff",
//                   fontSize: "15px", fontWeight: "800", letterSpacing: "2.5px",
//                   padding: "15px 40px", borderRadius: "50px",
//                   cursor: "pointer", width: "100%", maxWidth: "420px",
//                   transition: "all 0.3s ease",
//                 }}
//                 onMouseEnter={e => {
//                   e.currentTarget.style.background = "linear-gradient(135deg, #e63946, #c1121f)";
//                   e.currentTarget.style.boxShadow = "0 0 24px rgba(230,57,70,0.55)";
//                   e.currentTarget.style.transform = "translateY(-2px)";
//                 }}
//                 onMouseLeave={e => {
//                   e.currentTarget.style.background = "linear-gradient(135deg, #0f3460, #1a1a4e)";
//                   e.currentTarget.style.boxShadow = "none";
//                   e.currentTarget.style.transform = "translateY(0)";
//                 }}
//               >
//                 <span style={{
//                   width: "34px", height: "34px",
//                   background: "rgba(230,57,70,0.2)", borderRadius: "50%",
//                   display: "flex", alignItems: "center", justifyContent: "center",
//                 }}>
//                   <span style={{
//                     width: "13px", height: "13px",
//                     background: "#e63946", borderRadius: "50%",
//                     animation: "livePulse 1.2s infinite",
//                   }} />
//                 </span>
//                 GO LIVE
//               </button>


//               {/* Progress Bar */}
//               {progress > 0 && (
//                 <div className="w-[30%] text-center">
//                   <p className="text-gray-600 mb-1">
//                     {progress === 100
//                       ? "Video uploaded, processing... This may take 2+ minutes."
//                       : `Uploading ${progress}%...`}
//                   </p>
//                   {progress !== 100 && (
//                     <div className="w-full bg-gray-300 rounded-full h-4">
//                       <div
//                         className="bg-blue-500 h-4 rounded-full transition-all duration-200 ease-in-out"
//                         style={{ width: `${progress}%` }}
//                       ></div>
//                     </div>
//                   )}
//                 </div>
//               )}

//               {/* Output Video */}
//               {outputVideo && (
//                 <Dialog>
//                   <DialogTrigger asChild>
//                     <Button icon={videoViewIcon.src} onClick={handleStream}>
//                       View Processed Video
//                     </Button>
//                   </DialogTrigger>
//                   <DialogContent className="shad-dialog g7">
//                     <DialogHeader>
//                       <DialogTitle>Football Analytics Video</DialogTitle>
//                       <DialogDescription>
//                         You can share this with anyone who has this link.
//                       </DialogDescription>
//                     </DialogHeader>
//                     <video
//                       name="outputVideo"
//                       width={855}
//                       height={655}
//                       className="rounded-xl"
//                       autoPlay
//                       muted
//                       loop
//                       controls
//                       onError={(e) => {
//                         alert("Failed to load the video.");
//                         console.log("Video Error:", e.target.error);
//                       }}
//                     >
//                       <source id="video-source" src={videoApi} type="video/mp4" />
//                     </video>
//                     <div className="flex items-center space-x-2">
//                       <div className="grid flex-1 gap-2">
//                         <Label htmlFor="link" className="sr-only">Link</Label>
//                         <Input id="link" defaultValue={videoApi} readOnly className="g7" />
//                       </div>
//                       <button size="sm" className="px-3" aria-label="Copy link" onClick={handleCopyLink}>
//                         <Copy />
//                       </button>
//                     </div>
//                     <DialogFooter className="sm:justify-start">
//                       <button
//                         onClick={() => {
//                           if (outputVideo) {
//                             window.open(`${API_URL}/download/${outputVideo}`, "_blank");
//                           } else {
//                             alert("No video available for download.");
//                           }
//                         }}
//                         className="px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
//                       >
//                         DOWNLOAD
//                       </button>
//                     </DialogFooter>
//                   </DialogContent>
//                 </Dialog>
//               )}


//               <p className="text-sm text-gray-500">
//                 Video processing may take up to 2 minutes. Feel free to leave
//                 this page and come back later!
//               </p>
//             </div>
//           </div>
//         </div>
//       </Element>

//       {/* Login Modal */}
//       <LoginModal
//         isOpen={showLoginModal}
//         onClose={() => setShowLoginModal(false)}
//         onGoogleLoginSuccess={() => {
//           setIsAuthenticated(true);
//           handleUpload();
//         }}
//       />

//       {/* ── Live Tracking Modal ─────────────────────────────────────────── */}
//       {showLiveModal && (
//         <div
//           onClick={() => { if (!isStreaming) setShowLiveModal(false); }}
//           style={{
//             position: "fixed", inset: 0,
//             background: "rgba(0,0,0,0.88)",
//             display: "flex", alignItems: "center", justifyContent: "center",
//             zIndex: 9999, padding: "16px",
//           }}
//         >
//           <div
//             onClick={e => e.stopPropagation()}
//             style={{
//               background: "#0d1b2a", border: "1px solid #1e3a5f",
//               borderRadius: "16px", padding: "28px",
//               width: "100%", maxWidth: "820px",
//               maxHeight: "90vh", overflowY: "auto",
//               boxShadow: "0 0 50px rgba(0,150,255,0.12)",
//             }}
//           >
//             {/* Modal Header */}
//             <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "22px" }}>
//               <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
//                 {isStreaming && (
//                   <span style={{ color: "#e63946", fontWeight: 800, fontSize: "12px", letterSpacing: "2px", animation: "blink 1s infinite" }}>
//                     ● LIVE
//                   </span>
//                 )}
//                 <h2 style={{ color: "#fff", fontSize: "19px", fontWeight: 700, margin: 0 }}>
//                   Live Player &amp; Ball Tracking
//                 </h2>
//               </div>
//               {!isStreaming && (
//                 <button
//                   onClick={() => setShowLiveModal(false)}
//                   style={{
//                     background: "none", border: "1px solid #2a3a4a",
//                     color: "#888", width: "32px", height: "32px",
//                     borderRadius: "50%", cursor: "pointer", fontSize: "13px",
//                   }}
//                 >✕</button>
//               )}
//             </div>

//             {/* Before streaming */}
//             {!isStreaming && (
//               <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
//                 <label style={{ color: "#8899aa", fontSize: "13px", fontWeight: 600 }}>
//                   IPWebcam Stream URL
//                 </label>
//                 <input
//                   type="text"
//                   value={ipUrl}
//                   onChange={e => setIpUrl(e.target.value)}
//                   placeholder="http://192.168.x.x:8080/video"
//                   style={{
//                     width: "100%", padding: "12px 16px",
//                     background: "#132337", border: "1px solid #1e3a5f",
//                     borderRadius: "8px", color: "#fff",
//                     fontSize: "14px", boxSizing: "border-box",
//                   }}
//                 />
//                 {liveError && (
//                   <p style={{ color: "#e63946", fontSize: "13px", margin: 0 }}>{liveError}</p>
//                 )}
//                 <button
//                   onClick={startLive}
//                   disabled={liveLoading}
//                   style={{
//                     width: "100%", padding: "14px",
//                     background: "linear-gradient(135deg, #0096ff, #0052cc)",
//                     color: "#fff", fontWeight: 800, fontSize: "14px",
//                     letterSpacing: "2px", border: "none", borderRadius: "8px",
//                     cursor: liveLoading ? "not-allowed" : "pointer",
//                     opacity: liveLoading ? 0.6 : 1,
//                   }}
//                 >
//                   {liveLoading ? "Connecting..." : "▶  START LIVE TRACKING"}
//                 </button>

//                 {/* Instructions */}
//                 <div style={{ background: "#0a141e", border: "1px solid #1e3a5f", borderRadius: "8px", padding: "14px 16px" }}>
//                   <p style={{ color: "#8899aa", fontSize: "13px", fontWeight: 600, margin: "0 0 8px 0" }}>📱 How to set up:</p>
//                   <ol style={{ color: "#6677aa", fontSize: "13px", paddingLeft: "18px", margin: 0, lineHeight: 2 }}>
//                     <li>Install <strong style={{ color: "#aabbcc" }}>IP Webcam</strong> app on your Android phone</li>
//                     <li>Connect phone &amp; PC to the <strong style={{ color: "#aabbcc" }}>same WiFi</strong></li>
//                     <li>Open app → scroll down → tap <strong style={{ color: "#aabbcc" }}>Start Server</strong></li>
//                     <li>Copy the IP shown e.g. <code style={{ background: "#1a2a3a", padding: "1px 5px", borderRadius: "3px", color: "#66aaff", fontSize: "12px" }}>http://192.168.1.5:8080</code></li>
//                     <li>Paste above with <code style={{ background: "#1a2a3a", padding: "1px 5px", borderRadius: "3px", color: "#66aaff", fontSize: "12px" }}>/video</code> at the end → click Start</li>
//                   </ol>
//                 </div>
//               </div>
//             )}

//             {/* While streaming */}
//             {isStreaming && (
//               <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>

//                 {/* Live feed */}
//                 <div style={{ borderRadius: "10px", overflow: "hidden", border: "2px solid #e63946" }}>
//                   <img
//                     src={`${API_URL}/live/feed`}
//                     alt="Live tracking feed"
//                     style={{ width: "100%", display: "block" }}
//                   />
//                 </div>

//                 {/* Stats */}
//                 {trackingData && (
//                   <div style={{ background: "#0a141e", border: "1px solid #1e3a5f", borderRadius: "10px", padding: "16px" }}>
//                     <p style={{ color: "#facc15", fontWeight: 700, fontSize: "14px", margin: "0 0 12px 0" }}>Tracking Stats</p>

//                     <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
//                       <span style={{ color: "#8899aa", fontSize: "13px" }}>👤 Players detected</span>
//                       <span style={{ color: "#22c55e", fontWeight: 700 }}>{trackingData.player_count ?? 0}</span>
//                     </div>

//                     <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
//                       <span style={{ color: "#8899aa", fontSize: "13px" }}>⚽ Ball</span>
//                       <span style={{ color: trackingData.ball_detected ? "#facc15" : "#4b5563", fontWeight: 700 }}>
//                         {trackingData.ball_detected ? "Detected" : "Not found"}
//                       </span>
//                     </div>

//                     {trackingData.ball_detected && trackingData.ball_position && (
//                       <p style={{ color: "#4b6080", fontSize: "11px", margin: "0 0 8px 0" }}>
//                         Position: ({trackingData.ball_position[0]}, {trackingData.ball_position[1]}) | Conf: {trackingData.ball?.confidence}
//                       </p>
//                     )}

//                     {trackingData.referees?.length > 0 && (
//                       <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
//                         <span style={{ color: "#8899aa", fontSize: "13px" }}>🟠 Referee</span>
//                         <span style={{ color: "#fb923c", fontWeight: 700 }}>{trackingData.referees.length}</span>
//                       </div>
//                     )}

//                     <hr style={{ border: "none", borderTop: "1px solid #1e3a5f", margin: "10px 0" }} />

//                     {trackingData.players?.length > 0 && (
//                       <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
//                         {trackingData.players.map((pl, i) => (
//                           <div key={i} style={{ background: "#132337", color: "#8899aa", fontSize: "11px", padding: "4px 8px", borderRadius: "4px" }}>
//                             Player {i + 1} — pos: ({pl.center[0]}, {pl.center[1]}) — conf: {pl.confidence}
//                           </div>
//                         ))}
//                       </div>
//                     )}

//                     <p style={{ color: "#2a3a4a", fontSize: "11px", margin: "8px 0 0 0" }}>
//                       Total objects: {trackingData.total_objects}
//                     </p>
//                   </div>
//                 )}

//                 {/* Stop button */}
//                 <button
//                   onClick={stopLive}
//                   style={{
//                     width: "100%", padding: "13px",
//                     background: "#e63946", color: "#fff",
//                     fontWeight: 800, fontSize: "14px", letterSpacing: "2px",
//                     border: "none", borderRadius: "8px", cursor: "pointer",
//                   }}
//                   onMouseEnter={e => e.currentTarget.style.background = "#c1121f"}
//                   onMouseLeave={e => e.currentTarget.style.background = "#e63946"}
//                 >
//                   ⏹  STOP LIVE
//                 </button>
//               </div>
//             )}
//           </div>
//         </div>
//       )}

//       {/* Animations */}
//       <style>{`
//         @keyframes livePulse {
//           0%   { box-shadow: 0 0 0 0 rgba(230,57,70,0.7); }
//           70%  { box-shadow: 0 0 0 10px rgba(230,57,70,0); }
//           100% { box-shadow: 0 0 0 0 rgba(230,57,70,0); }
//         }
//         @keyframes blink {
//           0%, 100% { opacity: 1; }
//           50%       { opacity: 0.2; }
//         }
//       `}</style>
//     </section>
//   );
// };

// export default VideoUploader;

// import { useState, useEffect, useRef } from "react";
// import axios from "axios";
// import Dropzone from "react-dropzone";
// import "dropzone/dist/dropzone.css";
// import Button from "../components/Button";
// import { Element } from "react-scroll";
// import { API_URL } from "../api";
// import { Copy } from "lucide-react";
// import { useGoogleLogin } from "@react-oauth/google";

// import {
//   Dialog,
//   DialogContent,
//   DialogDescription,
//   DialogFooter,
//   DialogHeader,
//   DialogTitle,
//   DialogTrigger,
// } from "@/components/ui/dialog";
// import { Input } from "@/components/ui/input";
// import { Label } from "@/components/ui/label";

// import LoginModal from "../components/LoginModal.jsx";

// const VideoUploader = () => {
//   const [videoFile, setVideoFile] = useState(null);
//   const [outputVideo, setOutputVideo] = useState(null);
//   const [videoApi, setVideoApi] = useState(null);
//   const [progress, setProgress] = useState(0);
//   const [isAuthenticated, setIsAuthenticated] = useState(false);
//   const [showLoginModal, setShowLoginModal] = useState(false);

//   // Go Live state
//   const [showLiveModal, setShowLiveModal] = useState(false);
//   const [ipUrl, setIpUrl] = useState("http://192.168.1.5:8080/video");
//   const [isStreaming, setIsStreaming] = useState(false);
//   const [liveLoading, setLiveLoading] = useState(false);
//   const [liveError, setLiveError] = useState("");
//   const [trackingData, setTrackingData] = useState(null);
//   const dataInterval = useRef(null);

//   const buttonIcon = { src: "/images/plan-1.png", alt: "button Logo" };
//   const videoViewIcon = { src: "/images/plan-2.png", alt: "view video logo" };

//   // Poll /live/data every second while streaming
//   useEffect(() => {
//     if (isStreaming) {
//       dataInterval.current = setInterval(async () => {
//         try {
//           const res = await axios.get(`${API_URL}/live/data`);
//           setTrackingData(res.data);
//         } catch {}
//       }, 1000);
//     } else {
//       clearInterval(dataInterval.current);
//       setTrackingData(null);
//     }
//     return () => clearInterval(dataInterval.current);
//   }, [isStreaming]);

//   const handleCopyLink = () => {
//     navigator.clipboard.writeText(videoApi);
//     alert("Link copied to clipboard!");
//   };

//   const handleDrop = (acceptedFiles) => {
//     if (acceptedFiles.length > 0) {
//       setVideoFile(acceptedFiles[0]);
//     } else {
//       alert("Please drop a valid video file (mp4, avi, mov, mkv)");
//     }
//   };

//   const handleUpload = async () => {
//     if (!videoFile) return;
//     const formData = new FormData();
//     formData.append("file", videoFile);
//     try {
//       const uploadResponse = await axios.post(`${API_URL}/upload-video/`, formData, {
//         headers: { "Content-Type": "multipart/form-data" },
//         onUploadProgress: (event) => {
//           const percentCompleted = Math.round((event.loaded * 100) / event.total);
//           setProgress(percentCompleted);
//         },
//         timeout: 300000,
//       });
//       const videoUrl = uploadResponse.data.video_url;
//       const filename = videoUrl.split("/").pop();
//       setOutputVideo(filename);
//       alert("Video uploaded and processing started!");
//     } catch (error) {
//       console.error("Error uploading video", error);
//       alert("Failed to upload video. See console for details.");
//     } finally {
//       setProgress(0);
//     }
//   };

//   const handleStream = () => {
//     if (outputVideo) {
//       setVideoApi(`${API_URL}/output/${outputVideo}`);
//     }
//   };

//   const login = useGoogleLogin({
//     onSuccess: (credentialResponse) => {
//       setIsAuthenticated(true);
//       setShowLoginModal(false);
//       handleUpload();
//     },
//     onError: () => {
//       alert("Google login failed. Please try again.");
//     },
//   });

//   const handleUploadClick = () => {
//     if (!videoFile) {
//       alert("Please select a video file first!");
//       return;
//     }
//     if (!isAuthenticated) {
//       setShowLoginModal(true);
//     } else {
//       handleUpload();
//     }
//   };

//   // Live handlers
//   const handleGoLive = () => {
//     setLiveError("");
//     setShowLiveModal(true);
//   };

//   const startLive = async () => {
//     if (!ipUrl.trim()) {
//       setLiveError("Please enter your IPWebcam URL.");
//       return;
//     }
//     setLiveLoading(true);
//     setLiveError("");
//     try {
//       await axios.post(`${API_URL}/live/start?ip_url=${encodeURIComponent(ipUrl)}`);
//       setIsStreaming(true);
//     } catch {
//       setLiveError("Cannot connect. Check IP address and make sure phone & PC are on the same WiFi.");
//     } finally {
//       setLiveLoading(false);
//     }
//   };

//   const stopLive = async () => {
//     try {
//       await axios.post(`${API_URL}/live/stop`);
//     } catch {}
//     setIsStreaming(false);
//     setShowLiveModal(false);
//   };

//   return (
//     <section>
//       <Element name="create">
//         <div className="container">
//           <div className="flex flex-col items-center px-0 py-20 space-y-4 m-10 border-8 border-double border-[#34477C] rounded-2xl max-w-fit mx-auto">
//             <div className="flex flex-col gap-4 items-center justify-center mx-[190px]">
//               <h1 className="mb-2 h1 text-p4 uppercase max-lg:mb-2 max-lg:h2 max-md:mb-2 max-md:text-5xl max-md:leading-12">
//                 Make it happen!
//               </h1>
//               <p className="max-w-fit mb-6 body-1 max-md:mb-10 mx-auto">
//                 ⚽ Get your own video and realize its analytics. ⚽
//               </p>

//               {/* Dropzone */}
//               <div className="w-512 mx-auto">
//                 <Dropzone onDrop={handleDrop} accept={{ "video/*": [] }}>
//                   {({ getRootProps, getInputProps, isDragActive }) => (
//                     <div
//                       {...getRootProps()}
//                       className={`border-4 border-dashed rounded-md text-center cursor-pointer p-6 transition-all
//                         ${isDragActive ? "border-cyan-400 bg-cyan-50/10 scale-105" : "border-blue-500 hover:bg-blue-50/10"}`}
//                     >
//                       <input {...getInputProps()} />
//                       <p className="text-gray-600">
//                         {videoFile
//                           ? `✅ Selected: ${videoFile.name}`
//                           : isDragActive
//                           ? "Drop the video here..."
//                           : "Drag & Drop a video file here, or click to select"}
//                       </p>
//                     </div>
//                   )}
//                 </Dropzone>
//               </div>

//               {/* Upload Button */}
//               <Button
//                 onClick={handleUploadClick}
//                 disabled={!videoFile || progress > 0}
//                 icon={buttonIcon.src}
//               >
//                 {progress > 0 ? "Processing..." : "Upload and Process"}
//               </Button>

//               {/* ── GO LIVE Button — always visible below Upload ── */}
//               <button
//                 onClick={handleGoLive}
//                 style={{
//                   display: "flex", alignItems: "center", justifyContent: "center",
//                   gap: "12px",
//                   background: "linear-gradient(135deg, #0f3460, #1a1a4e)",
//                   border: "2px solid #e63946",
//                   color: "#ffffff",
//                   fontSize: "15px", fontWeight: "800", letterSpacing: "2.5px",
//                   padding: "15px 40px", borderRadius: "50px",
//                   cursor: "pointer", width: "100%", maxWidth: "420px",
//                   transition: "all 0.3s ease",
//                 }}
//                 onMouseEnter={e => {
//                   e.currentTarget.style.background = "linear-gradient(135deg, #e63946, #c1121f)";
//                   e.currentTarget.style.boxShadow = "0 0 24px rgba(230,57,70,0.55)";
//                   e.currentTarget.style.transform = "translateY(-2px)";
//                 }}
//                 onMouseLeave={e => {
//                   e.currentTarget.style.background = "linear-gradient(135deg, #0f3460, #1a1a4e)";
//                   e.currentTarget.style.boxShadow = "none";
//                   e.currentTarget.style.transform = "translateY(0)";
//                 }}
//               >
//                 <span style={{
//                   width: "34px", height: "34px",
//                   background: "rgba(230,57,70,0.2)", borderRadius: "50%",
//                   display: "flex", alignItems: "center", justifyContent: "center",
//                 }}>
//                   <span style={{
//                     width: "13px", height: "13px",
//                     background: "#e63946", borderRadius: "50%",
//                     animation: "livePulse 1.2s infinite",
//                   }} />
//                 </span>
//                 GO LIVE
//               </button>


//               {/* Progress Bar */}
//               {progress > 0 && (
//                 <div className="w-[30%] text-center">
//                   <p className="text-gray-600 mb-1">
//                     {progress === 100
//                       ? "Video uploaded, processing... This may take 2+ minutes."
//                       : `Uploading ${progress}%...`}
//                   </p>
//                   {progress !== 100 && (
//                     <div className="w-full bg-gray-300 rounded-full h-4">
//                       <div
//                         className="bg-blue-500 h-4 rounded-full transition-all duration-200 ease-in-out"
//                         style={{ width: `${progress}%` }}
//                       ></div>
//                     </div>
//                   )}
//                 </div>
//               )}

//               {/* Output Video */}
//               {outputVideo && (
//                 <Dialog>
//                   <DialogTrigger asChild>
//                     <Button icon={videoViewIcon.src} onClick={handleStream}>
//                       View Processed Video
//                     </Button>
//                   </DialogTrigger>
//                   <DialogContent className="shad-dialog g7">
//                     <DialogHeader>
//                       <DialogTitle>Football Analytics Video</DialogTitle>
//                       <DialogDescription>
//                         You can share this with anyone who has this link.
//                       </DialogDescription>
//                     </DialogHeader>
//                     <video
//                       name="outputVideo"
//                       width={855}
//                       height={655}
//                       className="rounded-xl"
//                       autoPlay
//                       muted
//                       loop
//                       controls
//                       onError={(e) => {
//                         alert("Failed to load the video.");
//                         console.log("Video Error:", e.target.error);
//                       }}
//                     >
//                       <source id="video-source" src={videoApi} type="video/mp4" />
//                     </video>
//                     <div className="flex items-center space-x-2">
//                       <div className="grid flex-1 gap-2">
//                         <Label htmlFor="link" className="sr-only">Link</Label>
//                         <Input id="link" defaultValue={videoApi} readOnly className="g7" />
//                       </div>
//                       <button size="sm" className="px-3" aria-label="Copy link" onClick={handleCopyLink}>
//                         <Copy />
//                       </button>
//                     </div>
//                     <DialogFooter className="sm:justify-start">
//                       <button
//                         onClick={() => {
//                           if (outputVideo) {
//                             window.open(`${API_URL}/download/${outputVideo}`, "_blank");
//                           } else {
//                             alert("No video available for download.");
//                           }
//                         }}
//                         className="px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
//                       >
//                         DOWNLOAD
//                       </button>
//                     </DialogFooter>
//                   </DialogContent>
//                 </Dialog>
//               )}


//               <p className="text-sm text-gray-500">
//                 Video processing may take up to 2 minutes. Feel free to leave
//                 this page and come back later!
//               </p>
//             </div>
//           </div>
//         </div>
//       </Element>

//       {/* Login Modal */}
//       <LoginModal
//         isOpen={showLoginModal}
//         onClose={() => setShowLoginModal(false)}
//         onGoogleLoginSuccess={() => {
//           setIsAuthenticated(true);
//           handleUpload();
//         }}
//       />

//       {/* ── Live Tracking Modal ─────────────────────────────────────────── */}
//       {showLiveModal && (
//         <div
//           onClick={() => { if (!isStreaming) setShowLiveModal(false); }}
//           style={{
//             position: "fixed", inset: 0,
//             background: "rgba(0,0,0,0.88)",
//             display: "flex", alignItems: "center", justifyContent: "center",
//             zIndex: 9999, padding: "16px",
//           }}
//         >
//           <div
//             onClick={e => e.stopPropagation()}
//             style={{
//               background: "#0d1b2a", border: "1px solid #1e3a5f",
//               borderRadius: "16px", padding: "28px",
//               width: "100%", maxWidth: "820px",
//               maxHeight: "90vh", overflowY: "auto",
//               boxShadow: "0 0 50px rgba(0,150,255,0.12)",
//             }}
//           >
//             {/* Modal Header */}
//             <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "22px" }}>
//               <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
//                 {isStreaming && (
//                   <span style={{ color: "#e63946", fontWeight: 800, fontSize: "12px", letterSpacing: "2px", animation: "blink 1s infinite" }}>
//                     ● LIVE
//                   </span>
//                 )}
//                 <h2 style={{ color: "#fff", fontSize: "19px", fontWeight: 700, margin: 0 }}>
//                   Live Player &amp; Ball Tracking
//                 </h2>
//               </div>
//               {!isStreaming && (
//                 <button
//                   onClick={() => setShowLiveModal(false)}
//                   style={{
//                     background: "none", border: "1px solid #2a3a4a",
//                     color: "#888", width: "32px", height: "32px",
//                     borderRadius: "50%", cursor: "pointer", fontSize: "13px",
//                   }}
//                 >✕</button>
//               )}
//             </div>

//             {/* Before streaming */}
//             {!isStreaming && (
//               <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
//                 <label style={{ color: "#8899aa", fontSize: "13px", fontWeight: 600 }}>
//                   IPWebcam Stream URL
//                 </label>
//                 <input
//                   type="text"
//                   value={ipUrl}
//                   onChange={e => setIpUrl(e.target.value)}
//                   placeholder="http://192.168.x.x:8080/video"
//                   style={{
//                     width: "100%", padding: "12px 16px",
//                     background: "#132337", border: "1px solid #1e3a5f",
//                     borderRadius: "8px", color: "#fff",
//                     fontSize: "14px", boxSizing: "border-box",
//                   }}
//                 />
//                 {liveError && (
//                   <p style={{ color: "#e63946", fontSize: "13px", margin: 0 }}>{liveError}</p>
//                 )}
//                 <button
//                   onClick={startLive}
//                   disabled={liveLoading}
//                   style={{
//                     width: "100%", padding: "14px",
//                     background: "linear-gradient(135deg, #0096ff, #0052cc)",
//                     color: "#fff", fontWeight: 800, fontSize: "14px",
//                     letterSpacing: "2px", border: "none", borderRadius: "8px",
//                     cursor: liveLoading ? "not-allowed" : "pointer",
//                     opacity: liveLoading ? 0.6 : 1,
//                   }}
//                 >
//                   {liveLoading ? "Connecting..." : "▶  START LIVE TRACKING"}
//                 </button>

//                 {/* Instructions */}
//                 <div style={{ background: "#0a141e", border: "1px solid #1e3a5f", borderRadius: "8px", padding: "14px 16px" }}>
//                   <p style={{ color: "#8899aa", fontSize: "13px", fontWeight: 600, margin: "0 0 8px 0" }}>📱 How to set up:</p>
//                   <ol style={{ color: "#6677aa", fontSize: "13px", paddingLeft: "18px", margin: 0, lineHeight: 2 }}>
//                     <li>Install <strong style={{ color: "#aabbcc" }}>IP Webcam</strong> app on your Android phone</li>
//                     <li>Connect phone &amp; PC to the <strong style={{ color: "#aabbcc" }}>same WiFi</strong></li>
//                     <li>Open app → scroll down → tap <strong style={{ color: "#aabbcc" }}>Start Server</strong></li>
//                     <li>Copy the IP shown e.g. <code style={{ background: "#1a2a3a", padding: "1px 5px", borderRadius: "3px", color: "#66aaff", fontSize: "12px" }}>http://192.168.1.5:8080</code></li>
//                     <li>Paste above with <code style={{ background: "#1a2a3a", padding: "1px 5px", borderRadius: "3px", color: "#66aaff", fontSize: "12px" }}>/video</code> at the end → click Start</li>
//                   </ol>
//                 </div>
//               </div>
//             )}

//             {/* While streaming */}
//             {isStreaming && (
//               <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>

//                 {/* Live feed */}
//                 <div style={{ borderRadius: "10px", overflow: "hidden", border: "2px solid #e63946" }}>
//                   <img
//                     src={`${API_URL}/live/feed`}
//                     alt="Live tracking feed"
//                     style={{ width: "100%", display: "block" }}
//                   />
//                 </div>

//                 {/* Stats */}
//                 {trackingData && (
//                   <div style={{ background: "#0a141e", border: "1px solid #1e3a5f", borderRadius: "10px", padding: "16px" }}>
//                     <p style={{ color: "#facc15", fontWeight: 700, fontSize: "14px", margin: "0 0 12px 0" }}>Tracking Stats</p>

//                     <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
//                       <span style={{ color: "#8899aa", fontSize: "13px" }}>👤 Players detected</span>
//                       <span style={{ color: "#22c55e", fontWeight: 700 }}>{trackingData.player_count ?? 0}</span>
//                     </div>

//                     <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
//                       <span style={{ color: "#8899aa", fontSize: "13px" }}>⚽ Ball</span>
//                       <span style={{ color: trackingData.ball_detected ? "#facc15" : "#4b5563", fontWeight: 700 }}>
//                         {trackingData.ball_detected ? "Detected" : "Not found"}
//                       </span>
//                     </div>

//                     {trackingData.ball_detected && trackingData.ball_position && (
//                       <p style={{ color: "#4b6080", fontSize: "11px", margin: "0 0 8px 0" }}>
//                         Position: ({trackingData.ball_position[0]}, {trackingData.ball_position[1]}) | Conf: {trackingData.ball?.confidence}
//                       </p>
//                     )}

//                     {trackingData.referees?.length > 0 && (
//                       <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
//                         <span style={{ color: "#8899aa", fontSize: "13px" }}>🟠 Referee</span>
//                         <span style={{ color: "#fb923c", fontWeight: 700 }}>{trackingData.referees.length}</span>
//                       </div>
//                     )}

//                     <hr style={{ border: "none", borderTop: "1px solid #1e3a5f", margin: "10px 0" }} />

//                     {trackingData.players?.length > 0 && (
//                       <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
//                         {trackingData.players.map((pl, i) => (
//                           <div key={i} style={{ background: "#132337", color: "#8899aa", fontSize: "11px", padding: "4px 8px", borderRadius: "4px" }}>
//                             Player {i + 1} — pos: ({pl.center[0]}, {pl.center[1]}) — conf: {pl.confidence}
//                           </div>
//                         ))}
//                       </div>
//                     )}

//                     <p style={{ color: "#2a3a4a", fontSize: "11px", margin: "8px 0 0 0" }}>
//                       Total objects: {trackingData.total_objects}
//                     </p>
//                   </div>
//                 )}

//                 {/* Stop button */}
//                 <button
//                   onClick={stopLive}
//                   style={{
//                     width: "100%", padding: "13px",
//                     background: "#e63946", color: "#fff",
//                     fontWeight: 800, fontSize: "14px", letterSpacing: "2px",
//                     border: "none", borderRadius: "8px", cursor: "pointer",
//                   }}
//                   onMouseEnter={e => e.currentTarget.style.background = "#c1121f"}
//                   onMouseLeave={e => e.currentTarget.style.background = "#e63946"}
//                 >
//                   ⏹  STOP LIVE
//                 </button>
//               </div>
//             )}
//           </div>
//         </div>
//       )}

//       {/* Animations */}
//       <style>{`
//         @keyframes livePulse {
//           0%   { box-shadow: 0 0 0 0 rgba(230,57,70,0.7); }
//           70%  { box-shadow: 0 0 0 10px rgba(230,57,70,0); }
//           100% { box-shadow: 0 0 0 0 rgba(230,57,70,0); }
//         }
//         @keyframes blink {
//           0%, 100% { opacity: 1; }
//           50%       { opacity: 0.2; }
//         }
//       `}</style>
//     </section>
//   );
// };

// export default VideoUploader;

import { useState, useEffect, useRef } from "react";
import axios from "axios";
import Dropzone from "react-dropzone";
import "dropzone/dist/dropzone.css";
import Button from "../components/Button";
import { Element } from "react-scroll";
import { API_URL } from "../api";
import { Copy } from "lucide-react";
import { useGoogleLogin } from "@react-oauth/google";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

import LoginModal from "../components/LoginModal.jsx";

const BACKEND_URL = "http://localhost:8000";

const VideoUploader = () => {
  const [videoFile, setVideoFile] = useState(null);
  const [outputVideo, setOutputVideo] = useState(null);
  const [videoApi, setVideoApi] = useState(null);
  const [progress, setProgress] = useState(0);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showLoginModal, setShowLoginModal] = useState(false);

  // Go Live state
  const [showLiveModal, setShowLiveModal] = useState(false);
  const [ipUrl, setIpUrl] = useState("http://192.168.1.5:8080/video");
  const [isStreaming, setIsStreaming] = useState(false);
  const [liveLoading, setLiveLoading] = useState(false);
  const [liveError, setLiveError] = useState("");
  const [trackingData, setTrackingData] = useState(null);
  const [feedKey, setFeedKey] = useState(0); // forces img reload
  const dataInterval = useRef(null);

  const buttonIcon = { src: "/images/plan-1.png", alt: "button Logo" };
  const videoViewIcon = { src: "/images/plan-2.png", alt: "view video logo" };

  // Poll /live/data every second while streaming
  useEffect(() => {
    if (isStreaming) {
      dataInterval.current = setInterval(async () => {
        try {
          const res = await axios.get(`${BACKEND_URL}/live/data`);
          setTrackingData(res.data);
        } catch {}
      }, 1000);
    } else {
      clearInterval(dataInterval.current);
      setTrackingData(null);
    }
    return () => clearInterval(dataInterval.current);
  }, [isStreaming]);

  const handleCopyLink = () => {
    navigator.clipboard.writeText(videoApi);
    alert("Link copied to clipboard!");
  };

  const handleDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setVideoFile(acceptedFiles[0]);
    } else {
      alert("Please drop a valid video file (mp4, avi, mov, mkv)");
    }
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    const formData = new FormData();
    formData.append("file", videoFile);
    try {
      const uploadResponse = await axios.post(`${API_URL}/upload-video/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (event) => {
          const percentCompleted = Math.round((event.loaded * 100) / event.total);
          setProgress(percentCompleted);
        },
        timeout: 300000,
      });
      const videoUrl = uploadResponse.data.video_url;
      const filename = videoUrl.split("/").pop();
      setOutputVideo(filename);
      alert("Video uploaded and processing started!");
    } catch (error) {
      console.error("Error uploading video", error);
      alert("Failed to upload video. See console for details.");
    } finally {
      setProgress(0);
    }
  };

  const handleStream = () => {
    if (outputVideo) {
      setVideoApi(`${API_URL}/output/${outputVideo}`);
    }
  };

  const login = useGoogleLogin({
    onSuccess: (credentialResponse) => {
      setIsAuthenticated(true);
      setShowLoginModal(false);
      handleUpload();
    },
    onError: () => {
      alert("Google login failed. Please try again.");
    },
  });

  const handleUploadClick = () => {
    if (!videoFile) {
      alert("Please select a video file first!");
      return;
    }
    if (!isAuthenticated) {
      setShowLoginModal(true);
    } else {
      handleUpload();
    }
  };

  // Live handlers
  const handleGoLive = () => {
    setLiveError("");
    setShowLiveModal(true);
  };

  const startLive = async () => {
    if (!ipUrl.trim()) {
      setLiveError("Please enter your IPWebcam URL.");
      return;
    }
    setLiveLoading(true);
    setLiveError("");
    try {
      await axios.post(`${BACKEND_URL}/live/start?ip_url=${encodeURIComponent(ipUrl)}`);
      setFeedKey(prev => prev + 1); // force img tag to reload fresh
      setIsStreaming(true);
    } catch {
      setLiveError("Cannot connect. Check IP address and make sure phone & PC are on the same WiFi.");
    } finally {
      setLiveLoading(false);
    }
  };

  const stopLive = async () => {
    try {
      await axios.post(`${BACKEND_URL}/live/stop`);
    } catch {}
    setIsStreaming(false);
    setShowLiveModal(false);
  };

  return (
    <section>
      <Element name="create">
        <div className="container">
          <div className="flex flex-col items-center px-0 py-20 space-y-4 m-10 border-8 border-double border-[#34477C] rounded-2xl max-w-fit mx-auto">
            <div className="flex flex-col gap-4 items-center justify-center mx-[190px]">
              <h1 className="mb-2 h1 text-p4 uppercase max-lg:mb-2 max-lg:h2 max-md:mb-2 max-md:text-5xl max-md:leading-12">
                Make it happen!
              </h1>
              <p className="max-w-fit mb-6 body-1 max-md:mb-10 mx-auto">
                ⚽ Get your own video and realize its analytics. ⚽
              </p>

              {/* Dropzone */}
              <div className="w-512 mx-auto">
                <Dropzone onDrop={handleDrop} accept={{ "video/*": [] }}>
                  {({ getRootProps, getInputProps, isDragActive }) => (
                    <div
                      {...getRootProps()}
                      className={`border-4 border-dashed rounded-md text-center cursor-pointer p-6 transition-all
                        ${isDragActive ? "border-cyan-400 bg-cyan-50/10 scale-105" : "border-blue-500 hover:bg-blue-50/10"}`}
                    >
                      <input {...getInputProps()} />
                      <p className="text-gray-600">
                        {videoFile
                          ? `✅ Selected: ${videoFile.name}`
                          : isDragActive
                          ? "Drop the video here..."
                          : "Drag & Drop a video file here, or click to select"}
                      </p>
                    </div>
                  )}
                </Dropzone>
              </div>

              {/* Upload Button */}
              <Button
                onClick={handleUploadClick}
                disabled={!videoFile || progress > 0}
                icon={buttonIcon.src}
              >
                {progress > 0 ? "Processing..." : "Upload and Process"}
              </Button>

              {/* GO LIVE Button */}
              <button
                onClick={handleGoLive}
                style={{
                  display: "flex", alignItems: "center", justifyContent: "center",
                  gap: "12px",
                  background: "linear-gradient(135deg, #0f3460, #1a1a4e)",
                  border: "2px solid #e63946",
                  color: "#ffffff",
                  fontSize: "15px", fontWeight: "800", letterSpacing: "2.5px",
                  padding: "15px 40px", borderRadius: "50px",
                  cursor: "pointer", width: "100%", maxWidth: "420px",
                  transition: "all 0.3s ease",
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = "linear-gradient(135deg, #e63946, #c1121f)";
                  e.currentTarget.style.boxShadow = "0 0 24px rgba(230,57,70,0.55)";
                  e.currentTarget.style.transform = "translateY(-2px)";
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = "linear-gradient(135deg, #0f3460, #1a1a4e)";
                  e.currentTarget.style.boxShadow = "none";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                <span style={{
                  width: "34px", height: "34px",
                  background: "rgba(230,57,70,0.2)", borderRadius: "50%",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <span style={{
                    width: "13px", height: "13px",
                    background: "#e63946", borderRadius: "50%",
                    animation: "livePulse 1.2s infinite",
                  }} />
                </span>
                GO LIVE
              </button>

              {/* Progress Bar */}
              {progress > 0 && (
                <div className="w-[30%] text-center">
                  <p className="text-gray-600 mb-1">
                    {progress === 100
                      ? "Video uploaded, processing... This may take 2+ minutes."
                      : `Uploading ${progress}%...`}
                  </p>
                  {progress !== 100 && (
                    <div className="w-full bg-gray-300 rounded-full h-4">
                      <div
                        className="bg-blue-500 h-4 rounded-full transition-all duration-200 ease-in-out"
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>
                  )}
                </div>
              )}

              {/* Output Video */}
              {outputVideo && (
                <Dialog>
                  <DialogTrigger asChild>
                    <Button icon={videoViewIcon.src} onClick={handleStream}>
                      View Processed Video
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="shad-dialog g7">
                    <DialogHeader>
                      <DialogTitle>Football Analytics Video</DialogTitle>
                      <DialogDescription>
                        You can share this with anyone who has this link.
                      </DialogDescription>
                    </DialogHeader>
                    <video
                      name="outputVideo"
                      width={855}
                      height={655}
                      className="rounded-xl"
                      autoPlay
                      muted
                      loop
                      controls
                      onError={(e) => {
                        alert("Failed to load the video.");
                        console.log("Video Error:", e.target.error);
                      }}
                    >
                      <source id="video-source" src={videoApi} type="video/mp4" />
                    </video>
                    <div className="flex items-center space-x-2">
                      <div className="grid flex-1 gap-2">
                        <Label htmlFor="link" className="sr-only">Link</Label>
                        <Input id="link" defaultValue={videoApi} readOnly className="g7" />
                      </div>
                      <button size="sm" className="px-3" aria-label="Copy link" onClick={handleCopyLink}>
                        <Copy />
                      </button>
                    </div>
                    <DialogFooter className="sm:justify-start">
                      <button
                        onClick={() => {
                          if (outputVideo) {
                            window.open(`${API_URL}/download/${outputVideo}`, "_blank");
                          } else {
                            alert("No video available for download.");
                          }
                        }}
                        className="px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                      >
                        DOWNLOAD
                      </button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              )}

              <p className="text-sm text-gray-500">
                Video processing may take up to 2 minutes. Feel free to leave
                this page and come back later!
              </p>
            </div>
          </div>
        </div>
      </Element>

      {/* Login Modal */}
      <LoginModal
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onGoogleLoginSuccess={() => {
          setIsAuthenticated(true);
          handleUpload();
        }}
      />

      {/* Live Tracking Modal */}
      {showLiveModal && (
        <div
          onClick={() => { if (!isStreaming) setShowLiveModal(false); }}
          style={{
            position: "fixed", inset: 0,
            background: "rgba(0,0,0,0.88)",
            display: "flex", alignItems: "center", justifyContent: "center",
            zIndex: 9999, padding: "16px",
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: "#0d1b2a", border: "1px solid #1e3a5f",
              borderRadius: "16px", padding: "28px",
              width: "100%", maxWidth: "900px",
              maxHeight: "90vh", overflowY: "auto",
              boxShadow: "0 0 50px rgba(0,150,255,0.12)",
            }}
          >
            {/* Modal Header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "22px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                {isStreaming && (
                  <span style={{ color: "#e63946", fontWeight: 800, fontSize: "12px", letterSpacing: "2px", animation: "blink 1s infinite" }}>
                    ● LIVE
                  </span>
                )}
                <h2 style={{ color: "#fff", fontSize: "19px", fontWeight: 700, margin: 0 }}>
                  Live Player &amp; Ball Tracking
                </h2>
              </div>
              {!isStreaming && (
                <button
                  onClick={() => setShowLiveModal(false)}
                  style={{
                    background: "none", border: "1px solid #2a3a4a",
                    color: "#888", width: "32px", height: "32px",
                    borderRadius: "50%", cursor: "pointer", fontSize: "13px",
                  }}
                >✕</button>
              )}
            </div>

            {/* Before streaming — IP input */}
            {!isStreaming && (
              <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                <label style={{ color: "#8899aa", fontSize: "13px", fontWeight: 600 }}>
                  IPWebcam Stream URL
                </label>
                <input
                  type="text"
                  value={ipUrl}
                  onChange={e => setIpUrl(e.target.value)}
                  placeholder="http://192.168.x.x:8080/video"
                  style={{
                    width: "100%", padding: "12px 16px",
                    background: "#132337", border: "1px solid #1e3a5f",
                    borderRadius: "8px", color: "#fff",
                    fontSize: "14px", boxSizing: "border-box",
                  }}
                />
                {liveError && (
                  <p style={{ color: "#e63946", fontSize: "13px", margin: 0 }}>{liveError}</p>
                )}
                <button
                  onClick={startLive}
                  disabled={liveLoading}
                  style={{
                    width: "100%", padding: "14px",
                    background: "linear-gradient(135deg, #0096ff, #0052cc)",
                    color: "#fff", fontWeight: 800, fontSize: "14px",
                    letterSpacing: "2px", border: "none", borderRadius: "8px",
                    cursor: liveLoading ? "not-allowed" : "pointer",
                    opacity: liveLoading ? 0.6 : 1,
                  }}
                >
                  {liveLoading ? "Connecting..." : "▶  START LIVE TRACKING"}
                </button>

                {/* Instructions */}
                <div style={{ background: "#0a141e", border: "1px solid #1e3a5f", borderRadius: "8px", padding: "14px 16px" }}>
                  <p style={{ color: "#8899aa", fontSize: "13px", fontWeight: 600, margin: "0 0 8px 0" }}>📱 How to set up:</p>
                  <ol style={{ color: "#6677aa", fontSize: "13px", paddingLeft: "18px", margin: 0, lineHeight: 2 }}>
                    <li>Install <strong style={{ color: "#aabbcc" }}>IP Webcam</strong> app on your Android phone</li>
                    <li>Connect phone &amp; PC to the <strong style={{ color: "#aabbcc" }}>same WiFi</strong></li>
                    <li>Open app → scroll down → tap <strong style={{ color: "#aabbcc" }}>Start Server</strong></li>
                    <li>Copy the IP shown on your phone screen</li>
                    <li>Paste above with <code style={{ background: "#1a2a3a", padding: "1px 5px", borderRadius: "3px", color: "#66aaff", fontSize: "12px" }}>/video</code> at end → click Start</li>
                  </ol>
                </div>
              </div>
            )}

            {/* While streaming — video feed + stats */}
            {isStreaming && (
              <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>

                {/* ── Live video feed ── */}
                <div style={{
                  borderRadius: "10px",
                  overflow: "hidden",
                  border: "2px solid #e63946",
                  background: "#000",
                  minHeight: "360px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  position: "relative",
                }}>
                  <img
                    key={feedKey}
                    src={`${BACKEND_URL}/live/feed?t=${feedKey}`}
                    alt="Live tracking feed"
                    style={{
                      width: "100%",
                      height: "auto",
                      display: "block",
                      minHeight: "360px",
                      objectFit: "contain",
                    }}
                    onError={(e) => {
                      console.error("Live feed failed to load");
                    }}
                  />
                  {/* Loading overlay text */}
                  <p style={{
                    position: "absolute",
                    color: "#333",
                    fontSize: "13px",
                    pointerEvents: "none",
                    zIndex: 0,
                  }}>
                    Connecting to feed...
                  </p>
                </div>

                {/* Stats panel */}
                {trackingData && (
                  <div style={{ background: "#0a141e", border: "1px solid #1e3a5f", borderRadius: "10px", padding: "16px" }}>
                    <p style={{ color: "#facc15", fontWeight: 700, fontSize: "14px", margin: "0 0 12px 0" }}>
                      Tracking Stats
                    </p>

                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                      <span style={{ color: "#8899aa", fontSize: "13px" }}>👤 Players detected</span>
                      <span style={{ color: "#22c55e", fontWeight: 700 }}>{trackingData.player_count ?? 0}</span>
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                      <span style={{ color: "#8899aa", fontSize: "13px" }}>⚽ Ball</span>
                      <span style={{ color: trackingData.ball_detected ? "#facc15" : "#4b5563", fontWeight: 700 }}>
                        {trackingData.ball_detected ? "Detected" : "Not found"}
                      </span>
                    </div>

                    {trackingData.ball_detected && trackingData.ball_position && (
                      <p style={{ color: "#4b6080", fontSize: "11px", margin: "0 0 8px 0" }}>
                        Position: ({trackingData.ball_position[0]}, {trackingData.ball_position[1]}) | Conf: {trackingData.ball?.confidence}
                      </p>
                    )}

                    {trackingData.referees?.length > 0 && (
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                        <span style={{ color: "#8899aa", fontSize: "13px" }}>🟠 Referee</span>
                        <span style={{ color: "#fb923c", fontWeight: 700 }}>{trackingData.referees.length}</span>
                      </div>
                    )}

                    <hr style={{ border: "none", borderTop: "1px solid #1e3a5f", margin: "10px 0" }} />

                    {trackingData.players?.length > 0 && (
                      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                        {trackingData.players.map((pl, i) => (
                          <div key={i} style={{ background: "#132337", color: "#8899aa", fontSize: "11px", padding: "4px 8px", borderRadius: "4px" }}>
                            Player {i + 1} — pos: ({pl.center[0]}, {pl.center[1]}) — conf: {pl.confidence}
                          </div>
                        ))}
                      </div>
                    )}

                    <p style={{ color: "#2a3a4a", fontSize: "11px", margin: "8px 0 0 0" }}>
                      Total objects: {trackingData.total_objects}
                    </p>
                  </div>
                )}

                {/* Stop button */}
                <button
                  onClick={stopLive}
                  style={{
                    width: "100%", padding: "13px",
                    background: "#e63946", color: "#fff",
                    fontWeight: 800, fontSize: "14px", letterSpacing: "2px",
                    border: "none", borderRadius: "8px", cursor: "pointer",
                    transition: "all 0.3s",
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = "#c1121f"}
                  onMouseLeave={e => e.currentTarget.style.background = "#e63946"}
                >
                  ⏹  STOP LIVE
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Animations */}
      <style>{`
        @keyframes livePulse {
          0%   { box-shadow: 0 0 0 0 rgba(230,57,70,0.7); }
          70%  { box-shadow: 0 0 0 10px rgba(230,57,70,0); }
          100% { box-shadow: 0 0 0 0 rgba(230,57,70,0); }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.2; }
        }
      `}</style>
    </section>
  );
};

export default VideoUploader;

