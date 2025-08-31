"""
Fixed demo with proper dimension handling
"""

import cv2
import numpy as np
import time
from realityguard_pro import RealityGuardPro, SafetyLevel

def create_demo_frame(height=720, width=1280):
    """Create a demo frame with specified dimensions"""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 60
    
    # Scale elements based on frame size
    scale = min(height/720, width/1280)
    
    # Add credit card mockup (scaled)
    card_w, card_h = int(350*scale), int(120*scale)
    card_x, card_y = int(100*scale), int(200*scale)
    
    cv2.rectangle(frame, (card_x, card_y), (card_x+card_w, card_y+card_h), (240, 240, 240), -1)
    cv2.rectangle(frame, (card_x, card_y), (card_x+card_w, card_y+card_h), (100, 100, 100), 2)
    
    font_scale = 0.8 * scale
    cv2.putText(frame, "4532 1234 5678 9012", (card_x+20, card_y+70), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), 2)
    
    # Add document (scaled)
    doc_x, doc_y = int(500*scale), int(150*scale)
    doc_w, doc_h = int(400*scale), int(300*scale)
    
    cv2.rectangle(frame, (doc_x, doc_y), (doc_x+doc_w, doc_y+doc_h), (255, 255, 255), -1)
    cv2.putText(frame, "CONFIDENTIAL", (doc_x+50, doc_y+40), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
    
    return frame

def run_demo():
    """Simplified demo that handles any camera resolution"""
    print("\n" + "="*70)
    print("REALITYGUARD PRO - SAFETY DEMO")
    print("="*70)
    
    system = RealityGuardPro(SafetyLevel.MODERATE)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    use_camera = cap.isOpened()
    
    if use_camera:
        # Get camera dimensions
        cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nCamera detected: {cam_width}x{cam_height}")
    else:
        cam_width, cam_height = 1280, 720
        print("\nUsing demo mode (no camera)")
    
    print("\nCONTROLS:")
    print("1-4: Change safety level")
    print("D: Toggle demo/camera")
    print("Q: Quit\n")
    
    show_demo = not use_camera
    fps_history = []
    
    while True:
        # Get frame
        if show_demo or not use_camera:
            frame = create_demo_frame(cam_height, cam_width)
        else:
            ret, frame = cap.read()
            if not ret:
                frame = create_demo_frame(cam_height, cam_width)
                show_demo = True
        
        # Process
        start = time.perf_counter()
        output, stats = system.process_frame(frame)
        process_time = (time.perf_counter() - start) * 1000
        
        fps = 1000 / process_time if process_time > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 60:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        
        # Draw HUD directly on output
        h, w = output.shape[:2]
        
        # Semi-transparent overlay for HUD
        overlay = output.copy()
        
        # Background for text
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
        
        # FPS
        fps_color = (0, 255, 0) if avg_fps >= 120 else (0, 150, 255)
        cv2.putText(output, f"FPS: {avg_fps:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, fps_color, 2)
        
        # Safety level
        cv2.putText(output, f"Safety: {stats['safety_level']}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detections
        if stats['safety_detections'] > 0:
            cv2.putText(output, f"Filtering: {stats['safety_detections']} items", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        # Mode indicator
        mode_text = "Camera" if not show_demo and use_camera else "Demo"
        cv2.putText(output, f"Mode: {mode_text}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Quest 3 ready
        if avg_fps >= 120:
            cv2.putText(output, "QUEST 3 READY", (w-200, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Instructions at bottom
        instructions = "1-4: Safety Level | D: Demo/Camera | Q: Quit"
        cv2.putText(output, instructions, (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show
        cv2.imshow('RealityGuard Pro', output)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            system.safety_level = SafetyLevel.FAMILY
            print("Safety: FAMILY")
        elif key == ord('2'):
            system.safety_level = SafetyLevel.MODERATE
            print("Safety: MODERATE")
        elif key == ord('3'):
            system.safety_level = SafetyLevel.WORKPLACE
            print("Safety: WORKPLACE")
        elif key == ord('4'):
            system.safety_level = SafetyLevel.UNRESTRICTED
            print("Safety: UNRESTRICTED")
        elif key == ord('d'):
            show_demo = not show_demo
            print(f"Switched to: {'Demo' if show_demo else 'Camera'}")
    
    if use_camera:
        cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nFinal Stats:")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Items filtered: {stats['filtered_total']}")

if __name__ == "__main__":
    run_demo()