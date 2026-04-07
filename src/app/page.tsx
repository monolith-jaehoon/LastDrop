"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./page.module.css";

type FaceLandmarkerInstance = import("@mediapipe/tasks-vision").FaceLandmarker;
type HandLandmarkerInstance = import("@mediapipe/tasks-vision").HandLandmarker;
type DrawingUtilsInstance = import("@mediapipe/tasks-vision").DrawingUtils;
type NormalizedLandmark = import("@mediapipe/tasks-vision").NormalizedLandmark;
type Category = import("@mediapipe/tasks-vision").Category;
type VisionModule = typeof import("@mediapipe/tasks-vision");
type HandOverlayState = {
  handednessLabel: string;
  isFist: boolean;
  landmarks: NormalizedLandmark[];
  wristAngle: number;
};

type CameraState = "idle" | "requesting" | "ready" | "unsupported" | "blocked";

const SAMPLE_WIDTH = 96;
const SAMPLE_HEIGHT = 54;
const ANALYSIS_INTERVAL_MS = 120;
const PIXEL_DELTA_THRESHOLD = 28;
const MOTION_HOLD_MS = 700;
const MIN_PLAYBACK_RATE = 0.45;
const MAX_PLAYBACK_RATE = 2.1;
const PLAYBACK_RATE_SMOOTHING = 0.26;
const PLAYBACK_RATE_MOTION_CAP = 18;
const MEDIAPIPE_WASM_PATH = "/mediapipe/wasm";
const FACE_LANDMARKER_MODEL_PATH = "/models/face_landmarker.task";
const HAND_LANDMARKER_MODEL_PATH = "/models/hand_landmarker.task";
const NOSE_BRIDGE_INDEX = 168;
const NOSE_TIP_INDEX = 1;

function getCameraErrorMessage(error: unknown) {
  if (!(error instanceof DOMException)) {
    return "카메라를 시작하지 못했습니다. 브라우저 권한과 연결 상태를 확인해 주세요.";
  }

  switch (error.name) {
    case "NotAllowedError":
      return "카메라 권한이 거부되었습니다. 주소창에서 허용한 뒤 다시 시도해 주세요.";
    case "NotFoundError":
      return "사용 가능한 카메라를 찾지 못했습니다.";
    case "NotReadableError":
      return "다른 앱이 이미 카메라를 사용 중일 수 있습니다.";
    default:
      return "카메라를 시작하지 못했습니다. HTTPS 또는 localhost 환경인지 확인해 주세요.";
  }
}

function getLandmarkPoint(
  landmarks: NormalizedLandmark[],
  index: number,
  width: number,
  height: number,
) {
  const landmark = landmarks[index];

  if (!landmark) {
    return null;
  }

  return {
    x: landmark.x * width,
    y: landmark.y * height,
  };
}

function drawLine(
  context: CanvasRenderingContext2D,
  start: { x: number; y: number } | null,
  end: { x: number; y: number } | null,
  color: string,
  lineWidth: number,
) {
  if (!start || !end) {
    return;
  }

  context.beginPath();
  context.moveTo(start.x, start.y);
  context.lineTo(end.x, end.y);
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.stroke();
}

function drawPoint(
  context: CanvasRenderingContext2D,
  point: { x: number; y: number } | null,
  color: string,
  radius: number,
) {
  if (!point) {
    return;
  }

  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fillStyle = color;
  context.shadowColor = color;
  context.shadowBlur = 12;
  context.fill();
  context.shadowBlur = 0;
}

function getDistanceBetweenLandmarks(
  start: NormalizedLandmark | undefined,
  end: NormalizedLandmark | undefined,
) {
  if (!start || !end) {
    return 0;
  }

  const dx = start.x - end.x;
  const dy = start.y - end.y;
  const dz = (start.z ?? 0) - (end.z ?? 0);

  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function getJointAngle(
  start: NormalizedLandmark | undefined,
  center: NormalizedLandmark | undefined,
  end: NormalizedLandmark | undefined,
) {
  if (!start || !center || !end) {
    return 180;
  }

  const ax = start.x - center.x;
  const ay = start.y - center.y;
  const az = (start.z ?? 0) - (center.z ?? 0);
  const bx = end.x - center.x;
  const by = end.y - center.y;
  const bz = (end.z ?? 0) - (center.z ?? 0);
  const magnitudeA = Math.sqrt(ax * ax + ay * ay + az * az);
  const magnitudeB = Math.sqrt(bx * bx + by * by + bz * bz);

  if (!magnitudeA || !magnitudeB) {
    return 180;
  }

  const cosine =
    (ax * bx + ay * by + az * bz) / (magnitudeA * magnitudeB);
  const clampedCosine = Math.min(1, Math.max(-1, cosine));

  return (Math.acos(clampedCosine) * 180) / Math.PI;
}

function normalizeAngle(angle: number) {
  return ((angle + 540) % 360) - 180;
}

function getWristAngle(landmarks: NormalizedLandmark[]) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];

  if (!wrist || !middleMcp) {
    return 0;
  }

  const rawAngle =
    (Math.atan2(middleMcp.y - wrist.y, middleMcp.x - wrist.x) * 180) / Math.PI +
    90;

  return normalizeAngle(rawAngle);
}

function isFingerCurled(
  landmarks: NormalizedLandmark[],
  mcpIndex: number,
  pipIndex: number,
  tipIndex: number,
) {
  return (
    getJointAngle(landmarks[mcpIndex], landmarks[pipIndex], landmarks[tipIndex]) <
    140
  );
}

function isThumbCurled(landmarks: NormalizedLandmark[]) {
  const thumbAngle = getJointAngle(landmarks[1], landmarks[2], landmarks[4]);
  const thumbToWrist = getDistanceBetweenLandmarks(landmarks[4], landmarks[0]);
  const thumbBaseToWrist = getDistanceBetweenLandmarks(landmarks[2], landmarks[0]);

  return thumbAngle < 145 || thumbToWrist < thumbBaseToWrist * 1.12;
}

function isFistPose(landmarks: NormalizedLandmark[]) {
  const curledFingerCount = [
    [5, 6, 8],
    [9, 10, 12],
    [13, 14, 16],
    [17, 18, 20],
  ].filter(([mcpIndex, pipIndex, tipIndex]) =>
    isFingerCurled(landmarks, mcpIndex, pipIndex, tipIndex),
  ).length;

  const palmSize = getDistanceBetweenLandmarks(landmarks[0], landmarks[9]) || 0.0001;
  const averageTipDistance =
    [4, 8, 12, 16, 20].reduce((sum, index) => {
      return sum + getDistanceBetweenLandmarks(landmarks[index], landmarks[0]);
    }, 0) / 5;

  return (
    curledFingerCount >= 4 &&
    (isThumbCurled(landmarks) || averageTipDistance < palmSize * 1.85)
  );
}

function getHandednessLabel(handedness: Category[] | undefined) {
  const label = handedness?.[0]?.categoryName;

  if (label === "Left") {
    return "왼손";
  }

  if (label === "Right") {
    return "오른손";
  }

  return "손";
}

function formatSignedAngle(angle: number) {
  const rounded = Math.round(angle);

  if (rounded > 0) {
    return `+${rounded}°`;
  }

  return `${rounded}°`;
}

function getPlaybackRateFromPixelChange(changedRatio: number) {
  const normalizedMotion = Math.min(
    Math.max(changedRatio, 0),
    PLAYBACK_RATE_MOTION_CAP,
  ) / PLAYBACK_RATE_MOTION_CAP;

  return (
    MIN_PLAYBACK_RATE +
    normalizedMotion * (MAX_PLAYBACK_RATE - MIN_PLAYBACK_RATE)
  );
}

function drawMirroredLabel(
  context: CanvasRenderingContext2D,
  canvasWidth: number,
  x: number,
  y: number,
  text: string,
  accentColor: string,
) {
  const fontSize = 13;
  const paddingX = 10;
  const boxHeight = 30;

  context.save();
  context.font = `600 ${fontSize}px sans-serif`;
  context.textBaseline = "middle";

  const textWidth = context.measureText(text).width;
  const boxWidth = textWidth + paddingX * 2;
  const labelX = Math.min(
    Math.max(x - boxWidth / 2, 12),
    canvasWidth - boxWidth - 12,
  );
  const labelY = Math.max(12, y - 42);

  context.scale(-1, 1);

  const mirroredBoxX = -labelX - boxWidth;

  context.fillStyle = "rgba(2, 6, 23, 0.78)";
  context.strokeStyle = accentColor;
  context.lineWidth = 1.5;
  context.fillRect(mirroredBoxX, labelY, boxWidth, boxHeight);
  context.strokeRect(mirroredBoxX, labelY, boxWidth, boxHeight);

  context.fillStyle = "#f8fafc";
  context.fillText(text, mirroredBoxX + paddingX, labelY + boxHeight / 2 + 0.5);
  context.restore();
}

export default function Home() {
  const stageRef = useRef<HTMLElement>(null);
  const sampleVideoRef = useRef<HTMLVideoElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const analysisContextRef = useRef<CanvasRenderingContext2D | null>(null);
  const overlayContextRef = useRef<CanvasRenderingContext2D | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameRequestRef = useRef<number | null>(null);
  const previousFrameRef = useRef<Uint8Array | null>(null);
  const lastAnalysisRef = useRef(0);
  const lastSignalAtRef = useRef<number | null>(null);
  const motionDetectedRef = useRef(false);
  const analysisActiveRef = useRef(false);
  const playbackRateRef = useRef(1);
  const sensitivityRef = useRef(14);
  const startCameraRef = useRef<() => Promise<void>>(async () => undefined);
  const faceLandmarkerRef = useRef<FaceLandmarkerInstance | null>(null);
  const faceLoadPromiseRef = useRef<Promise<FaceLandmarkerInstance> | null>(null);
  const handLandmarkerRef = useRef<HandLandmarkerInstance | null>(null);
  const handLoadPromiseRef = useRef<Promise<HandLandmarkerInstance> | null>(null);
  const faceModuleRef = useRef<VisionModule | null>(null);
  const drawingUtilsRef = useRef<DrawingUtilsInstance | null>(null);
  const lastFaceVideoTimeRef = useRef(-1);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [changedPixelRatio, setChangedPixelRatio] = useState(0);
  const [motionDetected, setMotionDetected] = useState(false);
  const [sensitivity, setSensitivity] = useState(14);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const prepareOverlayContext = () => {
    const canvas = overlayCanvasRef.current;
    const stage = stageRef.current;

    if (!canvas || !stage) {
      return null;
    }

    if (canvas.width !== stage.clientWidth || canvas.height !== stage.clientHeight) {
      canvas.width = stage.clientWidth;
      canvas.height = stage.clientHeight;
      overlayContextRef.current = canvas.getContext("2d");

      if (overlayContextRef.current && faceModuleRef.current) {
        drawingUtilsRef.current = new faceModuleRef.current.DrawingUtils(
          overlayContextRef.current,
        );
      }
    }

    const context = overlayContextRef.current ?? canvas.getContext("2d");

    if (!context) {
      return null;
    }

    overlayContextRef.current = context;
    context.clearRect(0, 0, canvas.width, canvas.height);

    return {
      context,
      width: canvas.width,
      height: canvas.height,
    };
  };

  const clearFaceOverlay = () => {
    const canvas = overlayCanvasRef.current;
    const context = overlayContextRef.current ?? canvas?.getContext("2d");

    if (!canvas || !context) {
      return;
    }

    overlayContextRef.current = context;
    context.clearRect(0, 0, canvas.width, canvas.height);
  };

  const drawTrackingOverlay = (
    faceLandmarksList: NormalizedLandmark[][],
    handStates: HandOverlayState[],
  ) => {
    const overlay = prepareOverlayContext();
    const vision = faceModuleRef.current;

    if (!overlay || !vision) {
      return;
    }

    const { context, width, height } = overlay;

    if (!drawingUtilsRef.current) {
      drawingUtilsRef.current = new vision.DrawingUtils(context);
    }

    const drawingUtils = drawingUtilsRef.current;

    if (!drawingUtils) {
      return;
    }

    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";

    for (const landmarks of faceLandmarksList) {
      drawingUtils.drawConnectors(
        landmarks,
        vision.FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        {
          color: "#7dd3fc",
          lineWidth: 2.5,
        },
      );
      drawingUtils.drawConnectors(
        landmarks,
        vision.FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        {
          color: "#7dd3fc",
          lineWidth: 2.5,
        },
      );
      drawingUtils.drawConnectors(
        landmarks,
        vision.FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        {
          color: "#67e8f9",
          lineWidth: 1.8,
        },
      );
      drawingUtils.drawConnectors(
        landmarks,
        vision.FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        {
          color: "#67e8f9",
          lineWidth: 1.8,
        },
      );
      drawingUtils.drawConnectors(
        landmarks,
        vision.FaceLandmarker.FACE_LANDMARKS_LIPS,
        {
          color: "#f9a8d4",
          lineWidth: 2.5,
        },
      );

      const noseBridge = getLandmarkPoint(
        landmarks,
        NOSE_BRIDGE_INDEX,
        width,
        height,
      );
      const noseTip = getLandmarkPoint(landmarks, NOSE_TIP_INDEX, width, height);

      drawLine(context, noseBridge, noseTip, "#fb923c", 3);
      drawPoint(context, noseBridge, "#fdba74", 4);
      drawPoint(context, noseTip, "#f97316", 6);
    }

    for (const handState of handStates) {
      const wrist = getLandmarkPoint(handState.landmarks, 0, width, height);
      const middleMcp = getLandmarkPoint(handState.landmarks, 9, width, height);
      const accentColor = handState.isFist ? "#fb7185" : "#22d3ee";
      const labelText = `${handState.handednessLabel} · ${
        handState.isFist ? "주먹" : "펴짐"
      } · ${formatSignedAngle(handState.wristAngle)}`;

      drawingUtils.drawConnectors(
        handState.landmarks,
        vision.HandLandmarker.HAND_CONNECTIONS,
        {
          color: accentColor,
          lineWidth: 2.6,
        },
      );
      drawingUtils.drawLandmarks(handState.landmarks, {
        color: "#f8fafc",
        fillColor: accentColor,
        lineWidth: 1,
        radius: 3.2,
      });

      drawLine(context, wrist, middleMcp, accentColor, 3);
      drawPoint(context, wrist, accentColor, 7);
      drawMirroredLabel(
        context,
        width,
        wrist?.x ?? 0,
        wrist?.y ?? 0,
        labelText,
        accentColor,
      );
    }

    context.restore();
  };

  const ensureFaceLandmarker = async () => {
    if (faceLandmarkerRef.current) {
      return faceLandmarkerRef.current;
    }

    if (faceLoadPromiseRef.current) {
      return faceLoadPromiseRef.current;
    }

    const promise = (async () => {
      const vision = faceModuleRef.current ?? (await import("@mediapipe/tasks-vision"));
      const wasmFileset = await vision.FilesetResolver.forVisionTasks(
        MEDIAPIPE_WASM_PATH,
      );
      const faceLandmarker = await vision.FaceLandmarker.createFromOptions(
        wasmFileset,
        {
          baseOptions: {
            modelAssetPath: FACE_LANDMARKER_MODEL_PATH,
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.6,
          minFacePresenceConfidence: 0.6,
          minTrackingConfidence: 0.5,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        },
      );

      faceModuleRef.current = vision;
      faceLandmarkerRef.current = faceLandmarker;

      if (overlayContextRef.current) {
        drawingUtilsRef.current = new vision.DrawingUtils(overlayContextRef.current);
      }

      return faceLandmarker;
    })().catch((error) => {
      faceLoadPromiseRef.current = null;
      console.error("Face landmarker init failed:", error);
      throw error;
    });

    faceLoadPromiseRef.current = promise;
    return promise;
  };

  const ensureHandLandmarker = async () => {
    if (handLandmarkerRef.current) {
      return handLandmarkerRef.current;
    }

    if (handLoadPromiseRef.current) {
      return handLoadPromiseRef.current;
    }

    const promise = (async () => {
      const vision = faceModuleRef.current ?? (await import("@mediapipe/tasks-vision"));
      const wasmFileset = await vision.FilesetResolver.forVisionTasks(
        MEDIAPIPE_WASM_PATH,
      );
      const handLandmarker = await vision.HandLandmarker.createFromOptions(
        wasmFileset,
        {
          baseOptions: {
            modelAssetPath: HAND_LANDMARKER_MODEL_PATH,
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.5,
        },
      );

      faceModuleRef.current = vision;
      handLandmarkerRef.current = handLandmarker;

      return handLandmarker;
    })().catch((error) => {
      handLoadPromiseRef.current = null;
      console.error("Hand landmarker init failed:", error);
      throw error;
    });

    handLoadPromiseRef.current = promise;
    return promise;
  };

  const queueNextAnalysis = () => {
    if (!analysisActiveRef.current) {
      return;
    }

    frameRequestRef.current = requestAnimationFrame(analyzeFrame);
  };

  const resetSampleVideo = () => {
    const sampleVideo = sampleVideoRef.current;

    if (!sampleVideo) {
      return;
    }

    sampleVideo.pause();
    sampleVideo.currentTime = 0;
    sampleVideo.playbackRate = 1;
    playbackRateRef.current = 1;
  };

  const syncSamplePlayback = (isActive: boolean, changedRatio: number) => {
    const sampleVideo = sampleVideoRef.current;

    if (!sampleVideo) {
      return;
    }

    if (!isActive) {
      sampleVideo.pause();
      sampleVideo.playbackRate = playbackRateRef.current;
      return;
    }

    const targetPlaybackRate = getPlaybackRateFromPixelChange(changedRatio);
    const nextPlaybackRate =
      playbackRateRef.current +
      (targetPlaybackRate - playbackRateRef.current) * PLAYBACK_RATE_SMOOTHING;

    playbackRateRef.current = nextPlaybackRate;
    sampleVideo.playbackRate = Number(nextPlaybackRate.toFixed(3));

    if (sampleVideo.paused) {
      void sampleVideo.play().catch(() => undefined);
    }
  };

  const stopCamera = (nextState: CameraState = "idle") => {
    analysisActiveRef.current = false;

    if (frameRequestRef.current !== null) {
      cancelAnimationFrame(frameRequestRef.current);
      frameRequestRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    previousFrameRef.current = null;
    analysisContextRef.current = null;
    lastAnalysisRef.current = 0;
    lastSignalAtRef.current = null;
    lastFaceVideoTimeRef.current = -1;
    motionDetectedRef.current = false;

    clearFaceOverlay();
    resetSampleVideo();

    setMotionDetected(false);
    setChangedPixelRatio(0);
    setCameraState(nextState);
  };

  const analyzeFrame = (timestamp: number) => {
    if (!analysisActiveRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = analysisCanvasRef.current;

    if (!video || !canvas) {
      queueNextAnalysis();
      return;
    }

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      queueNextAnalysis();
      return;
    }

    if (timestamp - lastAnalysisRef.current < ANALYSIS_INTERVAL_MS) {
      queueNextAnalysis();
      return;
    }

    lastAnalysisRef.current = timestamp;

    if (!analysisContextRef.current) {
      canvas.width = SAMPLE_WIDTH;
      canvas.height = SAMPLE_HEIGHT;
      analysisContextRef.current = canvas.getContext("2d", {
        willReadFrequently: true,
      });
    }

    const context = analysisContextRef.current;

    if (!context) {
      queueNextAnalysis();
      return;
    }

    context.drawImage(video, 0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);

    const frame = context.getImageData(0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);
    const pixelCount = SAMPLE_WIDTH * SAMPLE_HEIGHT;
    const currentFrame = new Uint8Array(pixelCount);
    const previousFrame = previousFrameRef.current;

    let totalDelta = 0;
    let changedPixels = 0;

    for (
      let pixelIndex = 0, dataIndex = 0;
      pixelIndex < pixelCount;
      pixelIndex += 1, dataIndex += 4
    ) {
      const grayscale =
        (frame.data[dataIndex] * 77 +
          frame.data[dataIndex + 1] * 150 +
          frame.data[dataIndex + 2] * 29) >>
        8;

      currentFrame[pixelIndex] = grayscale;

      if (!previousFrame) {
        continue;
      }

      const delta = Math.abs(grayscale - previousFrame[pixelIndex]);
      totalDelta += delta;

      if (delta > PIXEL_DELTA_THRESHOLD) {
        changedPixels += 1;
      }
    }

    if (!previousFrame) {
      setChangedPixelRatio(0);
      previousFrameRef.current = currentFrame;
      queueNextAnalysis();
      return;
    }

    const intensityScore = (totalDelta / pixelCount / 255) * 100;
    const changedRatio = (changedPixels / pixelCount) * 100;
    const combinedScore = Math.min(100, intensityScore * 0.6 + changedRatio * 1.8);
    const roundedRatio = Number(changedRatio.toFixed(1));
    const now = Date.now();

    setChangedPixelRatio(roundedRatio);

    const faceLandmarker = faceLandmarkerRef.current;
    const handLandmarker = handLandmarkerRef.current;
    let personVisible = false;

    if (
      (faceLandmarker || handLandmarker) &&
      video.currentTime !== lastFaceVideoTimeRef.current
    ) {
      lastFaceVideoTimeRef.current = video.currentTime;
      const faceLandmarksList = faceLandmarker
        ? faceLandmarker.detectForVideo(video, timestamp).faceLandmarks
        : [];
      const handResult = handLandmarker
        ? handLandmarker.detectForVideo(video, timestamp)
        : null;
      const handStates = handResult
        ? handResult.landmarks.map((landmarks, index) => {
            return {
              handednessLabel: getHandednessLabel(handResult.handedness[index]),
              isFist: isFistPose(landmarks),
              landmarks,
              wristAngle: getWristAngle(landmarks),
            };
          })
        : [];

      personVisible = faceLandmarksList.length > 0 || handStates.length > 0;
      drawTrackingOverlay(faceLandmarksList, handStates);
    } else if (!faceLandmarker && !handLandmarker) {
      clearFaceOverlay();
    }

    if (combinedScore >= sensitivityRef.current && personVisible) {
      lastSignalAtRef.current = now;
    }

    const isActive =
      lastSignalAtRef.current !== null &&
      now - lastSignalAtRef.current < MOTION_HOLD_MS;

    if (isActive !== motionDetectedRef.current) {
      motionDetectedRef.current = isActive;
      setMotionDetected(isActive);
    }

    syncSamplePlayback(isActive, changedRatio);

    previousFrameRef.current = currentFrame;
    queueNextAnalysis();
  };

  const startCamera = async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setCameraState("unsupported");
      setErrorMessage("이 브라우저에서는 웹캠 API를 사용할 수 없습니다.");
      return;
    }

    stopCamera("idle");
    setErrorMessage(null);
    setCameraState("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      const video = videoRef.current;

      if (!video) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("비디오 엘리먼트를 찾을 수 없습니다.");
      }

      streamRef.current = stream;
      video.srcObject = stream;
      video.muted = true;
      await video.play().catch(() => undefined);

      previousFrameRef.current = null;
      lastAnalysisRef.current = 0;
      lastSignalAtRef.current = null;
      lastFaceVideoTimeRef.current = -1;
      motionDetectedRef.current = false;
      analysisActiveRef.current = true;
      playbackRateRef.current = 1;

      setMotionDetected(false);
      setChangedPixelRatio(0);
      setCameraState("ready");

      void ensureFaceLandmarker().catch(() => undefined);
      void ensureHandLandmarker().catch(() => undefined);
      queueNextAnalysis();
    } catch (error) {
      stopCamera("blocked");
      setErrorMessage(getCameraErrorMessage(error));
    }
  };

  startCameraRef.current = startCamera;

  useEffect(() => {
    sensitivityRef.current = sensitivity;
  }, [sensitivity]);

  useEffect(() => {
    void startCameraRef.current();

    return () => {
      analysisActiveRef.current = false;

      if (frameRequestRef.current !== null) {
        cancelAnimationFrame(frameRequestRef.current);
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      drawingUtilsRef.current?.close();
      faceLandmarkerRef.current?.close();
      handLandmarkerRef.current?.close();
    };
  }, []);

  const showPlaceholder = cameraState !== "ready";

  return (
    <main className={styles.page}>
      <section ref={stageRef} className={styles.stage}>
        <video
          ref={sampleVideoRef}
          className={styles.video}
          src="/sample.mp4"
          muted
          loop
          preload="auto"
          playsInline
          aria-label="샘플 영상"
        />
        <video ref={videoRef} className={styles.cameraFeed} autoPlay muted playsInline />
        <canvas
          ref={overlayCanvasRef}
          className={styles.overlayCanvas}
          aria-hidden="true"
        />
        <div className={styles.hudTop}>
          <div
            className={`${styles.metricCard} ${
              motionDetected ? styles.metricCardActive : ""
            }`}
          >
            <span className={styles.metricLabel}>변화 픽셀 비율</span>
            <strong className={styles.metricValue}>{changedPixelRatio.toFixed(1)}%</strong>
          </div>
        </div>
        <div className={styles.hudBottom}>
          <div className={styles.sliderCard}>
            <div className={styles.sliderHeader}>
              <span className={styles.sliderLabel}>민감도 임계값</span>
              <strong className={styles.sliderValue}>{sensitivity.toFixed(0)}</strong>
            </div>
            <input
              className={styles.rangeInput}
              type="range"
              min="1"
              max="30"
              step="1"
              value={sensitivity}
              onChange={(event) => setSensitivity(Number(event.target.value))}
              aria-label="민감도 임계값"
            />
          </div>
        </div>
        {showPlaceholder && (
          <div className={styles.placeholder}>
            <strong>
              {cameraState === "requesting"
                ? "카메라 권한을 요청하고 있습니다."
                : "카메라를 사용할 수 없습니다."}
            </strong>
            <p>
              {errorMessage ??
                "권한을 허용하면 샘플 영상 위에 얼굴과 손 오버레이가 올라가고, 사람 움직임이 감지될 때만 샘플 영상이 재생됩니다."}
            </p>
            {cameraState !== "requesting" && (
              <button
                type="button"
                className={styles.retryButton}
                onClick={() => void startCamera()}
              >
                다시 시도
              </button>
            )}
          </div>
        )}
        <canvas ref={analysisCanvasRef} className={styles.hiddenCanvas} aria-hidden="true" />
      </section>
    </main>
  );
}
