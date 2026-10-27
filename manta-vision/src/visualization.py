import cv2
import numpy as np
from feature_matching import mono8


def draw(frame, det, algorithm, elapsed_ms, capture_fps):
    output = cv2.cvtColor(mono8(frame), cv2.COLOR_GRAY2BGR)
    if det is not None:
        box = np.rint(det.corners).astype(np.int32)
        cv2.polylines(output, [box], True, (40, 245, 55), 3)
        cx,cy = det.center
        cv2.circle(output,(round(cx),round(cy)),6,(0,0,255),-1)
        r = np.deg2rad(det.angle_deg)
        end = (round(cx+85*np.cos(r)),round(cy+85*np.sin(r)))
        cv2.arrowedLine(output,(round(cx),round(cy)),end,(255,0,255),3)
        status = 'COMPONENT DETECTED'
        info = f'Center: ({cx:.1f}, {cy:.1f}) px | Angle: {det.angle_deg:.1f} deg'
    else:
        status,info = 'NOT DETECTED', 'No reliable geometric match'
    # Readability with fixed background panel, not dependent on image brightness
    cv2.rectangle(output,(0,0),(min(output.shape[1]-1,825),146),(25,25,25),-1)
    cv2.putText(output, f'{algorithm} | {status}', (14,31),cv2.FONT_HERSHEY_SIMPLEX,.8,(80,230,255),2)
    cv2.putText(output,info,(14,64),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2)
    if det is not None:
        q = f'Inliers: {det.inliers}/{det.matches} | Error: {det.error_px:.2f}px | Scale: {det.scale:.2f}'
        cv2.putText(output,q,(14,93),cv2.FONT_HERSHEY_SIMPLEX,.57,(220,220,220),2)
    cv2.putText(output,f'Camera RX: {capture_fps:.1f} FPS | Detect: {elapsed_ms:.0f} ms',(14,124),cv2.FONT_HERSHEY_SIMPLEX,.58,(220,220,220),2)
    return output
