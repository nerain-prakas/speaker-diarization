"""
Production Cross-Language Voice Activity Detection System
Adaptive VAD for Hindi and Punjabi with optimized parameters
"""

import os
import numpy as np
import librosa
import json
from datetime import datetime
from scipy.signal import medfilt
from scipy.ndimage import binary_opening, binary_closing
import warnings
warnings.filterwarnings('ignore')

class AdaptiveCrossLanguageVAD:
    def __init__(self):
        self.sample_rate = 16000
        self.frame_length = 0.025  # 25ms frames
        self.frame_shift = 0.010   # 10ms shift
        
        # Optimized parameters from cross-language fine-tuning
        self.base_params = {
            'energy_threshold': 0.7845,
            'zcr_threshold': 0.6344,
            'spectral_threshold': 0.4629,
            'mfcc_weight': 0.3224,
            'contrast_weight': 0.3470
        }
        
        # Language-specific adjustments based on analysis
        self.language_adjustments = {
            'hindi': {
                'zcr_multiplier': 0.9,      # Hindi has lower ZCR
                'spectral_weight': 0.8,     # Less spectral emphasis
                'energy_boost': 1.0,        # Standard energy
                'frame_size_ms': 25         # Shorter frames for short files
            },
            'punjabi': {
                'zcr_multiplier': 1.15,     # Punjabi has higher ZCR
                'spectral_weight': 1.2,     # More spectral emphasis
                'energy_boost': 1.05,       # Slight energy boost
                'frame_size_ms': 40         # Longer frames for long files
            }
        }
        
        print("🌍 Adaptive Cross-Language VAD Initialized")
        print("🔧 Using optimized cross-language parameters")
        print("📊 Language-specific adaptations enabled")
    
    def detect_language(self, features):
        """Simple language detection based on key features"""
        duration = features['duration']
        spectral_centroid_mean = np.mean(features['spectral_centroid'])
        zcr_mean = np.mean(features['zcr'])
        
        # Simple heuristic based on our analysis
        # Punjabi: longer duration, higher spectral centroid, higher ZCR
        punjabi_score = 0
        
        if duration > 100:  # Punjabi files are typically much longer
            punjabi_score += 2
        elif duration > 10:
            punjabi_score += 1
        
        if spectral_centroid_mean > 1300:  # Punjabi has higher spectral centroid
            punjabi_score += 2
        elif spectral_centroid_mean > 1200:
            punjabi_score += 1
        
        if zcr_mean > 0.09:  # Punjabi has higher ZCR
            punjabi_score += 1
        
        # Decision
        if punjabi_score >= 3:
            return 'punjabi'
        else:
            return 'hindi'
    
    def extract_adaptive_features(self, audio_file, language=None):
        """Extract features with language-adaptive parameters"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Adaptive frame size based on language
            if language == 'punjabi':
                frame_length = int(0.040 * sr)  # 40ms for Punjabi
                hop_length = int(0.015 * sr)    # 15ms shift
            else:
                frame_length = int(0.025 * sr)  # 25ms for Hindi
                hop_length = int(0.010 * sr)    # 10ms shift
            
            # Core features
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Advanced features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
            
            # F0 estimation for pitch-aware processing
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            
            return {
                'energy': energy,
                'spectral_centroid': spectral_centroids,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'zcr': zcr,
                'mfccs': mfccs,
                'spectral_contrast': spectral_contrast,
                'f0': f0,
                'duration': len(y) / sr,
                'frames': len(energy),
                'sample_rate': sr
            }
        except Exception as e:
            print(f"❌ Error extracting features from {audio_file}: {e}")
            return None
    
    def adaptive_voice_activity_detection(self, features, language):
        """Language-adaptive VAD with optimized parameters"""
        if features is None:
            return np.array([])
        
        # Get base parameters
        base_energy_thresh = self.base_params['energy_threshold']
        base_zcr_thresh = self.base_params['zcr_threshold'] 
        base_spectral_thresh = self.base_params['spectral_threshold']
        mfcc_weight = self.base_params['mfcc_weight']
        contrast_weight = self.base_params['contrast_weight']
        
        # Apply language-specific adjustments
        lang_adj = self.language_adjustments[language]
        zcr_thresh = base_zcr_thresh * lang_adj['zcr_multiplier']
        spectral_weight = lang_adj['spectral_weight']
        energy_boost = lang_adj['energy_boost']
        
        # Extract and normalize features
        energy = features['energy'] * energy_boost
        zcr = features['zcr']
        spectral_centroid = features['spectral_centroid']
        mfcc_energy = np.mean(features['mfccs'][:3], axis=0)
        spectral_contrast = np.mean(features['spectral_contrast'], axis=0)
        f0 = features['f0']
        
        # Robust normalization
        def robust_normalize(x):
            q25, q75 = np.percentile(x, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                return np.clip((x - q25) / iqr, 0, 2)
            else:
                return np.ones_like(x)
        
        energy_norm = robust_normalize(energy)
        zcr_norm = robust_normalize(zcr)
        spectral_norm = robust_normalize(spectral_centroid)
        mfcc_norm = robust_normalize(mfcc_energy)
        contrast_norm = robust_normalize(spectral_contrast)
        
        # F0-based feature (voiced speech typically has stable F0)
        f0_stability = np.zeros_like(energy_norm)
        if len(f0) > 0:
            # Resample F0 to match other features
            if len(f0) != len(energy_norm):
                # Interpolate F0 to match feature length
                f0_indices = np.linspace(0, len(f0)-1, len(energy_norm))
                f0_resampled = np.interp(f0_indices, np.arange(len(f0)), f0)
            else:
                f0_resampled = f0
            
            valid_f0 = ~np.isnan(f0_resampled) & (f0_resampled > 50) & (f0_resampled < 500)
            if np.any(valid_f0):
                f0_smooth = medfilt(f0_resampled[valid_f0], kernel_size=min(5, np.sum(valid_f0)))
                f0_var = np.var(f0_smooth) if len(f0_smooth) > 1 else 0
                f0_stability[valid_f0] = 1.0 if f0_var < 1000 else 0.5
        
        # Multi-feature fusion with language-specific weighting
        speech_prob = (
            (energy_norm > base_energy_thresh).astype(float) * 0.25 +
            (zcr_norm < zcr_thresh).astype(float) * 0.20 +
            (spectral_norm > base_spectral_thresh).astype(float) * 0.15 * spectral_weight +
            mfcc_norm * mfcc_weight * 0.15 +
            contrast_norm * contrast_weight * 0.10 +
            f0_stability * 0.15  # Pitch-aware component
        )
        
        # Apply smoothing
        speech_prob = medfilt(speech_prob, kernel_size=min(7, len(speech_prob)))
        
        # Convert to binary decision
        vad_decision = (speech_prob > 0.45).astype(int)
        
        # Morphological operations for cleanup
        vad_decision = binary_closing(vad_decision, structure=np.ones(3))
        vad_decision = binary_opening(vad_decision, structure=np.ones(2))
        
        return vad_decision
    
    def process_audio_file(self, audio_file, language=None):
        """Process a single audio file with adaptive VAD"""
        print(f"🎵 Processing: {os.path.basename(audio_file)}")
        
        # Extract features
        features = self.extract_adaptive_features(audio_file, language)
        if features is None:
            return None
        
        # Auto-detect language if not specified
        if language is None:
            detected_language = self.detect_language(features)
            print(f"   🔍 Detected language: {detected_language}")
        else:
            detected_language = language
        
        # Apply adaptive VAD
        vad_result = self.adaptive_voice_activity_detection(features, detected_language)
        
        # Calculate statistics
        total_frames = len(vad_result)
        speech_frames = np.sum(vad_result)
        speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
        
        print(f"   📊 Language: {detected_language}")
        print(f"   ⏱️  Duration: {features['duration']:.2f}s")
        print(f"   🗣️  Speech frames: {speech_frames}/{total_frames} ({speech_ratio:.1%})")
        
        return {
            'filename': os.path.basename(audio_file),
            'language': detected_language,
            'duration': features['duration'],
            'total_frames': total_frames,
            'speech_frames': speech_frames,
            'speech_ratio': speech_ratio,
            'vad_result': vad_result,
            'features': features
        }
    
    def generate_rttm_file(self, results, output_file):
        """Generate RTTM file for speech segments"""
        with open(output_file, 'w') as f:
            for result in results:
                if result is None:
                    continue
                
                filename = result['filename']
                vad_result = result['vad_result']
                frame_shift = 0.015 if result['language'] == 'punjabi' else 0.010
                
                # Find speech segments
                speech_segments = []
                in_speech = False
                start_time = 0
                
                for i, is_speech in enumerate(vad_result):
                    time = i * frame_shift
                    
                    if is_speech and not in_speech:
                        # Start of speech segment
                        start_time = time
                        in_speech = True
                    elif not is_speech and in_speech:
                        # End of speech segment
                        duration = time - start_time
                        if duration > 0.1:  # Minimum segment length
                            speech_segments.append((start_time, duration))
                        in_speech = False
                
                # Handle case where speech continues to end
                if in_speech:
                    duration = len(vad_result) * frame_shift - start_time
                    if duration > 0.1:
                        speech_segments.append((start_time, duration))
                
                # Write RTTM entries
                for start, duration in speech_segments:
                    f.write(f"SPEAKER {filename} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker <NA> <NA>\n")
        
        print(f"📄 RTTM file saved: {output_file}")
    
    def batch_process(self, audio_directory, output_dir=None):
        """Process all audio files in a directory"""
        if output_dir is None:
            output_dir = audio_directory
        
        print(f"🚀 Batch Processing: {audio_directory}")
        print("=" * 60)
        
        audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]
        results = []
        
        for audio_file in audio_files:
            file_path = os.path.join(audio_directory, audio_file)
            result = self.process_audio_file(file_path)
            if result is not None:
                results.append(result)
        
        # Generate summary statistics
        if results:
            total_duration = sum(r['duration'] for r in results)
            total_speech_ratio = np.mean([r['speech_ratio'] for r in results])
            language_counts = {}
            for r in results:
                lang = r['language']
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            print(f"\n📈 Batch Processing Summary:")
            print(f"   📁 Files processed: {len(results)}")
            print(f"   ⏱️  Total duration: {total_duration:.1f}s")
            print(f"   🗣️  Average speech ratio: {total_speech_ratio:.1%}")
            print(f"   🌍 Language distribution: {language_counts}")
            
            # Generate RTTM file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rttm_file = os.path.join(output_dir, f"adaptive_vad_results_{timestamp}.rttm")
            self.generate_rttm_file(results, rttm_file)
            
            # Save detailed results
            results_file = os.path.join(output_dir, f"adaptive_vad_summary_{timestamp}.json")
            summary_data = {
                'timestamp': timestamp,
                'total_files': len(results),
                'total_duration': total_duration,
                'average_speech_ratio': total_speech_ratio,
                'language_distribution': language_counts,
                'file_results': [
                    {
                        'filename': r['filename'],
                        'language': r['language'],
                        'duration': r['duration'],
                        'speech_ratio': r['speech_ratio']
                    }
                    for r in results
                ]
            }
            
            with open(results_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"💾 Detailed results: {results_file}")
        
        return results


def main():
    """Main execution function"""
    print("🚀 Starting Adaptive Cross-Language VAD System")
    print("=" * 60)
    
    # Initialize VAD
    vad = AdaptiveCrossLanguageVAD()
    
    # Define dataset paths
    hindi_path = r"d:\codes\gen AI hackathon\segmentation\datasets\hindi-speech-recognition-dataset-main\hindi-speech-recognition-dataset-main\audio"
    punjabi_path = r"d:\codes\gen AI hackathon\segmentation\datasets\Panjabi-indicway2\audio"
    
    # Process Hindi dataset
    print("\n🇮🇳 Processing Hindi Dataset:")
    hindi_results = vad.batch_process(hindi_path)
    
    # Process Punjabi dataset
    print("\n🇮🇳 Processing Punjabi Dataset:")
    punjabi_results = vad.batch_process(punjabi_path)
    
    # Combined analysis
    all_results = hindi_results + punjabi_results
    
    if all_results:
        print("\n🌍 Cross-Language Analysis Complete!")
        print(f"📊 Total files processed: {len(all_results)}")
        print(f"🇮🇳 Hindi files: {len(hindi_results)}")
        print(f"🇮🇳 Punjabi files: {len(punjabi_results)}")
        
        # Language-specific performance
        for language in ['hindi', 'punjabi']:
            lang_results = [r for r in all_results if r['language'] == language]
            if lang_results:
                avg_speech_ratio = np.mean([r['speech_ratio'] for r in lang_results])
                print(f"   {language.title()} average speech ratio: {avg_speech_ratio:.1%}")
    
    return vad, all_results


if __name__ == "__main__":
    main()