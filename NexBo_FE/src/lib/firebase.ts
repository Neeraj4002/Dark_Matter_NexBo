import { initializeApp, getApps } from 'firebase/app'
import { 
  getAuth, 
  GoogleAuthProvider,
  signInWithPopup,
  sendPasswordResetEmail,
  sendEmailVerification,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  onAuthStateChanged,
  signOut,
  type User
} from 'firebase/auth'

// Your Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyCbIFQ9aoS2BAnez30K7EncnE9DIBkqD00",
  authDomain: "nexbo-17952.firebaseapp.com",
  projectId: "nexbo-17952",
  storageBucket: "nexbo-17952.firebasestorage.app", 
  messagingSenderId: "334271960698",
  appId: "1:334271960698:web:a54fc9fc51f088a4410dc3",
  measurementId: "G-294F8G2J0L"
}

// Initialize Firebase
const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0]
const auth = getAuth(app)
const googleProvider = new GoogleAuthProvider()

// Helper functions
const signInWithGoogle = async () => {
  try {
    const result = await signInWithPopup(auth, googleProvider)
    return { user: result.user, error: null }
  } catch (error: any) {
    return { user: null, error: error.message }
  }
}

const resetPassword = async (email: string) => {
  try {
    await sendPasswordResetEmail(auth, email)
    return { error: null }
  } catch (error: any) {
    return { error: error.message }
  }
}

const verifyEmail = async (user: User) => {
  try {
    await sendEmailVerification(user)
    return { error: null }
  } catch (error: any) {
    return { error: error.message }
  }
}

export { 
  auth, 
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  onAuthStateChanged,
  signOut,
  signInWithGoogle,
  resetPassword,
  verifyEmail
}