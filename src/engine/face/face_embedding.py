import cv2
import numpy as np
import logging
from deepface import DeepFace

logger = logging.getLogger(__name__)

class FaceEmbedder:
    def __init__(self, model_name: str = "ArcFace"):
        """
        Initialize the Face Embedder with the specified model.
        
        Args:
            model_name (str): The name of the face recognition model to use.
                              Options: "VGG-Face", "Facenet", "Facenet512", "OpenFace", 
                                       "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace".
                              Defaults to "ArcFace".
        """
        self.model_name = model_name
        # Trigger weight download on initialization if not present
        try:
             # Just checking if we can build the model or if it needs download
             # DeepFace.build_model(model_name) 
             # Actually, DeepFace downloads lazily when represent/verify is called.
             # We can do a dummy call or just log.
             logger.info(f"FaceEmbedder initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize FaceEmbedder: {e}")
            raise

    def get_embedding(self, img_path_or_array):
        """
        Get the face embedding for a single face in the image.
        
        Args:
            img_path_or_array (str or numpy.ndarray): Path to the image or the image array (BGR if cv2).
        
        Returns:
            list: The embedding vector.
            None: If no face is detected or multiple faces are found (depending on config, but here just robustly).
        """
        try:
            # enforce_detection=False allows getting embedding even if face detection is weak, 
            # but usually we want to ensure there is a face.
            # However, in a proctoring scenario, we might have already detected a face with MediaPipe.
            # DeepFace has its own detectors. Let's use retinaface or opencv (default) backend.
            embeddings = DeepFace.represent(
                img_path=img_path_or_array,
                model_name=self.model_name,
                enforce_detection=True, # Ensure a face is present
                detector_backend="opencv" # Fast
            )
            
            if embeddings:
                return embeddings[0]["embedding"]
            return None
            
        except ValueError as e:
            logger.warning(f"Face detection failed during embedding generation: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def verify_identity(self, img1_path_or_array, img2_path_or_array, threshold: float = None) -> bool:
        """
        Verify if two images belong to the same person.
        
        Args:
            img1_path_or_array: First image (e.g., reference ID card/profile photo).
            img2_path_or_array: Second image (e.g., current webcam frame).
            threshold (float): Optional custom threshold. If None, uses model default.
            
        Returns:
            bool: True if verified, False otherwise.
        """
        try:
            result = DeepFace.verify(
                img1_path=img1_path_or_array,
                img2_path=img2_path_or_array,
                model_name=self.model_name,
                detector_backend="opencv",
                enforce_detection=False # Sometimes reference image might be cropped or different
            )
            
            # If threshold is provided, check distance manually? 
            # DeepFace result["verified"] already uses a tuned threshold.
            if threshold is not None:
                return result["distance"] < threshold
                
            return result["verified"]
            
        except Exception as e:
            logger.error(f"Error verification identity: {e}")
            return False

    def compare_embeddings(self, emb1, emb2, threshold: float = 0.68):
        """
        Compare two embeddings using Cosine Similarity (actually Distance).
        Threshold for ArcFace is typically 0.68 (Cosine Distance).
        
        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.
            threshold: Distance threshold. If distance < threshold, it's a match.
        
        Returns:
            (bool, float): (is_match, distance)
        """
        if emb1 is None or emb2 is None:
            return False, 1.0 # Max distance
            
        try:
            # Cosine Distance = 1 - Cosine Similarity
            # Similarity = dot(a, b) / (norm(a) * norm(b))
            
            a = np.array(emb1)
            b = np.array(emb2)
            
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            cosine_similarity = dot_product / (norm_a * norm_b)
            cosine_distance = 1 - cosine_similarity
            
            return cosine_distance < threshold, cosine_distance
            
        except Exception as e:
             logger.error(f"Error comparing embeddings: {e}")
             return False, 1.0
