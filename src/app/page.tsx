"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./page.module.css";

type FaceLandmarkerInstance = import("@mediapipe/tasks-vision").FaceLandmarker;
type HandLandmarkerInstance = import("@mediapipe/tasks-vision").HandLandmarker;
type NormalizedLandmark = import("@mediapipe/tasks-vision").NormalizedLandmark;
type Category = import("@mediapipe/tasks-vision").Category;
type VisionModule = typeof import("@mediapipe/tasks-vision");
type GameMode = "normal" | "hard";
type Point = {
  x: number;
  y: number;
};
type Circle = Point & {
  radius: number;
};
type SceneObject = {
  alpha: number;
  depth: number;
  directionX: number;
  directionY: number;
  id: number;
  rotation: number;
  size: number;
  spin: number;
  sway: number;
};
type HandOverlayState = {
  handednessLabel: string;
  isFist: boolean;
  landmarks: NormalizedLandmark[];
  wristAngle: number;
};
type ArmPose = {
  elbow: Point;
  hand: Point;
  handRadius: number;
  isFist: boolean;
  shoulder: Point;
  wristAngle: number;
};
type PlayerPose = {
  arms: {
    left: ArmPose;
    right: ArmPose;
  };
  head: Circle;
  hitZones: Circle[];
  hips: {
    left: Point;
    right: Point;
  };
  neck: {
    height: number;
    width: number;
  } & Point;
  shoulders: {
    left: Point;
    right: Point;
  };
  torso: {
    center: Point;
    height: number;
    width: number;
  };
};
type SceneObjectProjection = {
  impactRadius: number;
  impactX: number;
  impactY: number;
  opacity: number;
  scale: number;
  spriteSize: number;
  x: number;
  y: number;
};
type CupFillState = {
  left: number;
  right: number;
};
type HardModeResult = {
  beerPercent: number;
  fatiguePercent: number;
  grade: string;
  message: string;
  remainingPercent: number;
  scorePercent: number;
  subtitle: string;
};
type DebugOverlayState = {
  faceLandmarks: NormalizedLandmark[] | null;
  handStates: HandOverlayState[];
};
type DebugVisibility = {
  eyes: boolean;
  handJoints: boolean;
  mouth: boolean;
  nose: boolean;
};

type CameraState = "idle" | "requesting" | "ready" | "unsupported" | "blocked";
type PlayerSpriteKey =
  | "body"
  | "head"
  | "leftArm"
  | "leftHand"
  | "rightArm"
  | "rightHand";
type PlayerSpriteMap = Record<PlayerSpriteKey, HTMLImageElement | null>;
type SpriteSegmentAnchors = {
  end: Point;
  start: Point;
};

const SAMPLE_WIDTH = 96;
const SAMPLE_HEIGHT = 54;
const ANALYSIS_INTERVAL_MS = 120;
const PIXEL_DELTA_THRESHOLD = 28;
const MOTION_HOLD_MS = 220;
const PLAYER_POSE_HOLD_MS = 360;
const PLAYER_DAMAGE_COOLDOWN_MS = 280;
const PLAYER_HIT_FLASH_MS = 200;
const PLAYER_MAX_HP = 100;
const PLAYER_TORSO_ALPHA = 1;
const PLAYER_HEAD_ALPHA = 1;
const PLAYER_ARM_ALPHA = 1;
const PLAYER_HAND_ALPHA = 1;
const BEER_DAMAGE = 8;
const BEER_START_DELAY_MS = 5000;
const BASE_SCENE_OBJECT_COUNT = 14;
const DEFAULT_BEER_DENSITY = 0.1;
const MIN_BEER_DENSITY = 0.1;
const MAX_BEER_DENSITY = 2;
const DEFAULT_GOAL_DISTANCE_METERS = 30;
const MIN_GOAL_DISTANCE_METERS = 10;
const MAX_GOAL_DISTANCE_METERS = 60;
const SCENE_FRAME_INTERVAL_MS = 1000 / 30;
const SCENE_MIN_SPEED = 0.16;
const SCENE_MAX_SPEED = 1.06;
const SCENE_MOTION_CAP = 18;
const LEFT_CUP_SAFE_MIN_ANGLE = 15;
const LEFT_CUP_SAFE_MAX_ANGLE = 25;
const RIGHT_CUP_SAFE_MIN_ANGLE = -25;
const RIGHT_CUP_SAFE_MAX_ANGLE = -15;
const CUP_SPILL_FULL_TILT_MARGIN = 45;
const CUP_SPILL_RATE = 0.03;
const CUP_FILL_WARNING_RATIO = 0.35;
const PUBLIC_BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const withBasePath = (assetPath: string) => `${PUBLIC_BASE_PATH}${assetPath}`;
const MEDIAPIPE_WASM_PATH = withBasePath("/mediapipe/wasm");
const FACE_LANDMARKER_MODEL_PATH = withBasePath("/models/face_landmarker.task");
const HAND_LANDMARKER_MODEL_PATH = withBasePath("/models/hand_landmarker.task");
const BEER_SPRITE_PATH = withBasePath("/images/beer-stein.svg");
const BACKDROP_VIDEO_PATH = withBasePath("/bg.mp4");
const PLAYER_SPRITE_PATHS = {
  body: withBasePath("/images/body.png"),
  head: withBasePath("/images/head.png"),
  leftArm: withBasePath("/images/left_arm.png"),
  leftHand: withBasePath("/images/left_hand.png"),
  rightArm: withBasePath("/images/right_arm.png"),
  rightHand: withBasePath("/images/right_hand.png"),
} satisfies Record<PlayerSpriteKey, string>;
const BODY_SPRITE_SEGMENT_ANCHORS: SpriteSegmentAnchors = {
  end: { x: 0.5, y: 0.68 },
  start: { x: 0.5, y: 0.06 },
};
const HEAD_SPRITE_SEGMENT_ANCHORS: SpriteSegmentAnchors = {
  end: { x: 0.5, y: 0.93 },
  start: { x: 0.5, y: 0.02 },
};
const LEFT_ARM_SPRITE_SEGMENT_ANCHORS: SpriteSegmentAnchors = {
  end: { x: 0.66, y: 0.06 },
  start: { x: 0.33, y: 0.89 },
};
const RIGHT_ARM_SPRITE_SEGMENT_ANCHORS: SpriteSegmentAnchors = {
  end: { x: 0.33, y: 0.08 },
  start: { x: 0.64, y: 0.94 },
};
const LEFT_HAND_SPRITE_SEGMENT_ANCHORS: SpriteSegmentAnchors = {
  end: { x: 0.68, y: 0.68 },
  start: { x: 0.54, y: 0.05 },
};
const RIGHT_HAND_SPRITE_SEGMENT_ANCHORS: SpriteSegmentAnchors = {
  end: { x: 0.32, y: 0.68 },
  start: { x: 0.52, y: 0.07 },
};
const SCENE_LANE_X = [-0.52, -0.34, -0.18, -0.06, 0.06, 0.18, 0.34, 0.52];
const SCENE_LANE_Y = [-0.28, -0.12, 0.02, 0.18, 0.3];
const SCENE_OBJECT_HALO_OFFSET_Y_RATIO = -0.3;
const SCENE_OBJECT_HALO_RADIUS_X_RATIO = 0.76;
const SCENE_OBJECT_HALO_RADIUS_Y_RATIO = 0.9;
const SCENE_OBJECT_IMPACT_OFFSET_X_RATIO = 0;
const SCENE_OBJECT_IMPACT_OFFSET_Y_RATIO = -0.3;
const SCENE_OBJECT_IMPACT_RADIUS_RATIO = 0.15;
const FACE_LEFT_EYE_INDICES = [33, 133, 159, 145];
const FACE_RIGHT_EYE_INDICES = [362, 263, 386, 374];
const FACE_NOSE_INDICES = [1, 4, 168];
const FACE_MOUTH_INDICES = [13, 14, 78, 308];
const HAND_DEBUG_LABELS = [
  "WRIST",
  "THUMB_CMC",
  "THUMB_MCP",
  "THUMB_IP",
  "THUMB_TIP",
  "INDEX_MCP",
  "INDEX_PIP",
  "INDEX_DIP",
  "INDEX_TIP",
  "MIDDLE_MCP",
  "MIDDLE_PIP",
  "MIDDLE_DIP",
  "MIDDLE_TIP",
  "RING_MCP",
  "RING_PIP",
  "RING_DIP",
  "RING_TIP",
  "PINKY_MCP",
  "PINKY_PIP",
  "PINKY_DIP",
  "PINKY_TIP",
] as const;
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
] as const;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

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

function drawLine(
  context: CanvasRenderingContext2D,
  start: Point,
  end: Point,
  color: string,
  lineWidth: number,
) {
  context.beginPath();
  context.moveTo(start.x, start.y);
  context.lineTo(end.x, end.y);
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.stroke();
}

function drawPoint(
  context: CanvasRenderingContext2D,
  point: Point,
  color: string,
  radius: number,
) {
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fillStyle = color;
  context.fill();
}

function drawRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const clampedRadius = Math.min(radius, width / 2, height / 2);

  context.beginPath();
  context.moveTo(x + clampedRadius, y);
  context.lineTo(x + width - clampedRadius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + clampedRadius);
  context.lineTo(x + width, y + height - clampedRadius);
  context.quadraticCurveTo(x + width, y + height, x + width - clampedRadius, y + height);
  context.lineTo(x + clampedRadius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - clampedRadius);
  context.lineTo(x, y + clampedRadius);
  context.quadraticCurveTo(x, y, x + clampedRadius, y);
  context.closePath();
}

function createEmptyPlayerSprites(): PlayerSpriteMap {
  return {
    body: null,
    head: null,
    leftArm: null,
    leftHand: null,
    rightArm: null,
    rightHand: null,
  };
}

function drawSegmentSprite(
  context: CanvasRenderingContext2D,
  image: HTMLImageElement | null,
  start: Point,
  end: Point,
  anchors: SpriteSegmentAnchors,
) {
  if (!image) {
    return false;
  }

  const imageWidth = image.naturalWidth || image.width;
  const imageHeight = image.naturalHeight || image.height;

  if (!imageWidth || !imageHeight) {
    return false;
  }

  const sourceStart = {
    x: imageWidth * anchors.start.x,
    y: imageHeight * anchors.start.y,
  };
  const sourceEnd = {
    x: imageWidth * anchors.end.x,
    y: imageHeight * anchors.end.y,
  };
  const sourceDx = sourceEnd.x - sourceStart.x;
  const sourceDy = sourceEnd.y - sourceStart.y;
  const sourceLength = Math.hypot(sourceDx, sourceDy);
  const destinationDx = end.x - start.x;
  const destinationDy = end.y - start.y;
  const destinationLength = Math.hypot(destinationDx, destinationDy);

  if (!sourceLength || !destinationLength) {
    return false;
  }

  context.save();
  context.translate(start.x, start.y);
  context.rotate(
    Math.atan2(destinationDy, destinationDx) - Math.atan2(sourceDy, sourceDx),
  );
  context.scale(destinationLength / sourceLength, destinationLength / sourceLength);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, -sourceStart.x, -sourceStart.y, imageWidth, imageHeight);
  context.restore();

  return true;
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

  const cosine = (ax * bx + ay * by + az * bz) / (magnitudeA * magnitudeB);
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

function createFullCupFill(): CupFillState {
  return {
    left: 1,
    right: 1,
  };
}

function getCombinedCupFill(cupFill: CupFillState) {
  return (cupFill.left + cupFill.right) / 2;
}

function getCupSafeRange(side: "left" | "right") {
  if (side === "left") {
    return {
      max: LEFT_CUP_SAFE_MAX_ANGLE,
      min: LEFT_CUP_SAFE_MIN_ANGLE,
    };
  }

  return {
    max: RIGHT_CUP_SAFE_MAX_ANGLE,
    min: RIGHT_CUP_SAFE_MIN_ANGLE,
  };
}

function getCupSafeCenter(side: "left" | "right") {
  const range = getCupSafeRange(side);

  return (range.min + range.max) / 2;
}

function getCupUnsafeDelta(side: "left" | "right", wristAngle: number) {
  const range = getCupSafeRange(side);

  if (wristAngle < range.min) {
    return range.min - wristAngle;
  }

  if (wristAngle > range.max) {
    return wristAngle - range.max;
  }

  return 0;
}

function formatSignedAngle(angle: number) {
  const rounded = Math.round(angle);

  if (rounded > 0) {
    return `+${rounded}°`;
  }

  return `${rounded}°`;
}

function getCupTiltSeverity(side: "left" | "right", wristAngle: number) {
  return clamp(
    getCupUnsafeDelta(side, wristAngle) / CUP_SPILL_FULL_TILT_MARGIN,
    0,
    1,
  );
}

function getHardModeResult(
  cupFill: CupFillState,
  remainingFatigueRatio: number,
): HardModeResult {
  const beerRatio = getCombinedCupFill(cupFill);
  const fatigueRatio = clamp(remainingFatigueRatio, 0, 1);
  const scoreRatio = (beerRatio + fatigueRatio) / 2;
  const remainingPercent = Math.round(beerRatio * 100);
  const fatiguePercent = Math.round(fatigueRatio * 100);
  const scorePercent = Math.round(scoreRatio * 100);

  if (scoreRatio >= 0.9) {
    return {
      beerPercent: remainingPercent,
      fatiguePercent,
      grade: "S",
      message:
        "맥주도 손목도 체력도 거의 완벽했습니다. 축제 본부에서 바로 명예 브루마스터 배지를 달아줄 분위기예요.",
      remainingPercent,
      scorePercent,
      subtitle: "브루마스터 오브 레전드",
    };
  }

  if (scoreRatio >= 0.75) {
    return {
      beerPercent: remainingPercent,
      fatiguePercent,
      grade: "A",
      message:
        "맥주와 피로도를 모두 훌륭하게 관리했습니다. 주변 테이블에서 건배 타이밍을 맞춰 따라오고 있어요.",
      remainingPercent,
      scorePercent,
      subtitle: "황금 손목 챔피언",
    };
  }

  if (scoreRatio >= 0.55) {
    return {
      beerPercent: remainingPercent,
      fatiguePercent,
      grade: "B",
      message:
        "조금 지치긴 했지만 리듬은 잘 지켰습니다. 축제 MC가 다음 라운드 참가를 권하고 있어요.",
      remainingPercent,
      scorePercent,
      subtitle: "거품 수호대",
    };
  }

  if (scoreRatio >= 0.35) {
    return {
      beerPercent: remainingPercent,
      fatiguePercent,
      grade: "C",
      message:
        "맥주와 피로도 모두 아슬아슬했지만 완주는 해냈습니다. 밴드가 엔딩 드럼롤만큼은 길게 쳐주고 있어요.",
      remainingPercent,
      scorePercent,
      subtitle: "간신히 건배 성공",
    };
  }

  return {
    beerPercent: remainingPercent,
    fatiguePercent,
    grade: "D",
    message:
      "끝까지 버틴 것만으로도 충분히 드라마틱했습니다. 테이블 아래서 무릎은 떨렸어도 클리어는 클리어예요.",
    remainingPercent,
    scorePercent,
    subtitle: "거품만 남은 생존자",
  };
}

function getSceneSpeedFromPixelChange(changedRatio: number) {
  const normalizedMotion =
    clamp(changedRatio, 0, SCENE_MOTION_CAP) / SCENE_MOTION_CAP;

  return SCENE_MIN_SPEED + normalizedMotion * (SCENE_MAX_SPEED - SCENE_MIN_SPEED);
}

function getSceneObjectCount(beerDensity: number) {
  return Math.max(1, Math.round(BASE_SCENE_OBJECT_COUNT * beerDensity));
}

function formatDistanceMeters(ms: number) {
  return `${(ms / 1000).toFixed(1)}m`;
}

function getMirroredLandmarkPoint(
  landmark: NormalizedLandmark | undefined,
  width: number,
  height: number,
) {
  if (!landmark) {
    return null;
  }

  return {
    x: width - landmark.x * width,
    y: landmark.y * height,
  };
}

function getAverageMirroredLandmarkPoint(
  landmarks: NormalizedLandmark[],
  indices: number[],
  width: number,
  height: number,
) {
  let totalX = 0;
  let totalY = 0;
  let count = 0;

  for (const index of indices) {
    const point = getMirroredLandmarkPoint(landmarks[index], width, height);

    if (!point) {
      continue;
    }

    totalX += point.x;
    totalY += point.y;
    count += 1;
  }

  if (!count) {
    return null;
  }

  return {
    x: totalX / count,
    y: totalY / count,
  };
}

function drawDebugTag(
  context: CanvasRenderingContext2D,
  point: Point,
  label: string,
  color: string,
  offsetY = -18,
) {
  const fontSize = 11;
  const paddingX = 7;
  const paddingY = 4;
  context.save();
  context.font = `600 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  const textWidth = context.measureText(label).width;
  const tagWidth = textWidth + paddingX * 2;
  const tagHeight = fontSize + paddingY * 2;
  const tagX = point.x - tagWidth / 2;
  const tagY = point.y + offsetY - tagHeight / 2;

  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 1.25;
  context.beginPath();
  context.moveTo(point.x, point.y - 2);
  context.lineTo(point.x, point.y + offsetY * 0.55);
  context.stroke();
  drawRoundedRect(context, tagX, tagY, tagWidth, tagHeight, 8);
  context.fillStyle = "rgba(2, 6, 23, 0.88)";
  context.fill();
  context.stroke();
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(label, point.x, tagY + tagHeight / 2 + 0.5);
  context.restore();
}

function drawFaceDebugOverlay(
  context: CanvasRenderingContext2D,
  faceLandmarks: NormalizedLandmark[],
  width: number,
  height: number,
  visibility: DebugVisibility,
) {
  context.save();
  context.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";

  if (visibility.eyes) {
    const leftEyePoint = getAverageMirroredLandmarkPoint(
      faceLandmarks,
      FACE_LEFT_EYE_INDICES,
      width,
      height,
    );
    const rightEyePoint = getAverageMirroredLandmarkPoint(
      faceLandmarks,
      FACE_RIGHT_EYE_INDICES,
      width,
      height,
    );

    if (leftEyePoint) {
      drawPoint(context, leftEyePoint, "rgba(96, 165, 250, 0.95)", 4);
      drawDebugTag(context, leftEyePoint, "LEFT EYE", "rgba(96, 165, 250, 0.95)");
    }

    if (rightEyePoint) {
      drawPoint(context, rightEyePoint, "rgba(96, 165, 250, 0.95)", 4);
      drawDebugTag(context, rightEyePoint, "RIGHT EYE", "rgba(96, 165, 250, 0.95)");
    }
  }

  if (visibility.nose) {
    const nosePoint = getAverageMirroredLandmarkPoint(
      faceLandmarks,
      FACE_NOSE_INDICES,
      width,
      height,
    );

    if (nosePoint) {
      drawPoint(context, nosePoint, "rgba(250, 204, 21, 0.95)", 4);
      drawDebugTag(context, nosePoint, "NOSE", "rgba(250, 204, 21, 0.95)", -22);
    }
  }

  if (visibility.mouth) {
    const mouthPoint = getAverageMirroredLandmarkPoint(
      faceLandmarks,
      FACE_MOUTH_INDICES,
      width,
      height,
    );

    if (mouthPoint) {
      drawPoint(context, mouthPoint, "rgba(244, 114, 182, 0.95)", 4);
      drawDebugTag(context, mouthPoint, "MOUTH", "rgba(244, 114, 182, 0.95)", 22);
    }
  }

  context.restore();
}

function drawHandDebugOverlay(
  context: CanvasRenderingContext2D,
  handState: HandOverlayState,
  width: number,
  height: number,
) {
  const baseColor =
    handState.handednessLabel === "왼손"
      ? "rgba(56, 189, 248, 0.95)"
      : "rgba(251, 146, 60, 0.95)";
  const points = handState.landmarks.map((landmark) =>
    getMirroredLandmarkPoint(landmark, width, height),
  );

  context.save();
  context.strokeStyle = baseColor;
  context.lineWidth = 1.5;

  for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
    const start = points[startIndex];
    const end = points[endIndex];

    if (!start || !end) {
      continue;
    }

    drawLine(context, start, end, "rgba(226, 232, 240, 0.35)", 1.5);
  }

  points.forEach((point, index) => {
    if (!point) {
      return;
    }

    drawPoint(context, point, baseColor, 3.4);
    drawDebugTag(
      context,
      point,
      `${index} ${HAND_DEBUG_LABELS[index]}`,
      baseColor,
      index % 2 === 0 ? -16 : 16,
    );
  });

  const wristPoint = points[0];

  if (wristPoint) {
    drawDebugTag(
      context,
      wristPoint,
      `${handState.handednessLabel} ${formatSignedAngle(handState.wristAngle)}`,
      baseColor,
      -34,
    );
  }

  context.restore();
}

function drawDebugOverlay(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  debugOverlay: DebugOverlayState | null,
  visibility: DebugVisibility,
) {
  if (!debugOverlay) {
    return;
  }

  if (
    debugOverlay.faceLandmarks &&
    (visibility.eyes || visibility.nose || visibility.mouth)
  ) {
    drawFaceDebugOverlay(
      context,
      debugOverlay.faceLandmarks,
      width,
      height,
      visibility,
    );
  }

  if (visibility.handJoints) {
    for (const handState of debugOverlay.handStates) {
      drawHandDebugOverlay(context, handState, width, height);
    }
  }
}

function getMirroredFaceBounds(
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
) {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const landmark of landmarks) {
    const point = getMirroredLandmarkPoint(landmark, width, height);

    if (!point) {
      continue;
    }

    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }

  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
    return null;
  }

  return {
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    height: maxY - minY,
    width: maxX - minX,
  };
}

function createSceneObject(id: number, depth = 0.02 + Math.random() * 0.1): SceneObject {
  const laneX = SCENE_LANE_X[Math.floor(Math.random() * SCENE_LANE_X.length)] ?? 0;
  const laneY = SCENE_LANE_Y[Math.floor(Math.random() * SCENE_LANE_Y.length)] ?? 0;

  return {
    alpha: 0.55 + Math.random() * 0.35,
    depth,
    directionX: clamp(laneX + (Math.random() - 0.5) * 0.1, -0.62, 0.62),
    directionY: clamp(laneY + (Math.random() - 0.5) * 0.14, -0.36, 0.38),
    id,
    rotation: Math.random() * Math.PI * 2,
    size: 18 + Math.random() * 16,
    spin: (0.08 + Math.random() * 0.32) * (Math.random() > 0.5 ? 1 : -1),
    sway: 0.7 + Math.random() * 1.2,
  };
}

function createSceneObjects(count: number, pattern: "staggered" | "flow" = "flow") {
  return Array.from({ length: count }, (_, index) => {
    if (pattern === "staggered") {
      const progress = count <= 1 ? 0 : index / (count - 1);
      const depth = 0.006 + Math.pow(progress, 1.8) * 0.26;

      return createSceneObject(index, depth);
    }

    return createSceneObject(index, 0.08 + (index / count) * 0.92);
  });
}

function getSceneObjectProjection(
  object: SceneObject,
  width: number,
  height: number,
): SceneObjectProjection {
  const centerX = width / 2;
  const centerY = height / 2;
  const perspective = Math.pow(object.depth, 1.56);
  const spread = 0.08 + perspective * 0.92;
  const x = centerX + object.directionX * width * 0.36 * spread;
  const y =
    centerY +
    object.directionY * height * 0.34 * spread +
    Math.sin(object.rotation * 0.7) * object.sway * perspective * 12;
  const scale = object.size * (0.55 + perspective * 3.3);
  const opacity = clamp(object.alpha * (0.24 + perspective * 1.2), 0.08, 1);
  const spriteSize = scale * 2.45;

  return {
    // Match hits to the stein body center instead of the decorative glow below it.
    impactRadius: Math.max(12, spriteSize * SCENE_OBJECT_IMPACT_RADIUS_RATIO),
    impactX: x + spriteSize * SCENE_OBJECT_IMPACT_OFFSET_X_RATIO,
    impactY: y + spriteSize * SCENE_OBJECT_IMPACT_OFFSET_Y_RATIO,
    opacity,
    scale,
    spriteSize,
    x,
    y,
  };
}

function drawSceneObject(
  context: CanvasRenderingContext2D,
  projection: SceneObjectProjection,
  rotation: number,
  beerSprite: HTMLImageElement | null,
) {
  context.save();
  context.translate(projection.x, projection.y);
  context.rotate(rotation);
  context.globalAlpha = projection.opacity * 0.26;
  context.fillStyle = "rgba(54, 20, 7, 0.9)";
  context.beginPath();
  context.ellipse(
    0,
    projection.scale * 0.7,
    projection.scale * 0.6,
    projection.scale * 0.16,
    0,
    0,
    Math.PI * 2,
  );
  context.fill();

  context.globalAlpha = projection.opacity * 0.14;
  context.fillStyle = "rgba(255, 203, 116, 1)";
  context.beginPath();
  context.ellipse(
    0,
    projection.spriteSize * SCENE_OBJECT_HALO_OFFSET_Y_RATIO,
    projection.scale * SCENE_OBJECT_HALO_RADIUS_X_RATIO,
    projection.scale * SCENE_OBJECT_HALO_RADIUS_Y_RATIO,
    0,
    0,
    Math.PI * 2,
  );
  context.fill();

  context.globalAlpha = projection.opacity;

  if (beerSprite) {
    context.drawImage(
      beerSprite,
      -projection.spriteSize / 2,
      -projection.spriteSize * 0.86,
      projection.spriteSize,
      projection.spriteSize,
    );
  } else {
    context.fillStyle = "rgba(249, 168, 37, 0.9)";
    context.strokeStyle = "rgba(255, 243, 191, 0.85)";
    context.lineWidth = Math.max(1, projection.scale * 0.08);
    drawRoundedRect(
      context,
      -projection.scale * 0.42,
      -projection.scale * 0.82,
      projection.scale * 0.84,
      projection.scale * 1.24,
      projection.scale * 0.18,
    );
    context.fill();
    context.stroke();
    context.beginPath();
    context.arc(
      0,
      -projection.scale * 0.8,
      projection.scale * 0.45,
      Math.PI,
      Math.PI * 2,
    );
    context.stroke();
  }

  context.restore();
}

function drawBeerCup(
  context: CanvasRenderingContext2D,
  arm: ArmPose,
  side: "left" | "right",
  headRadius: number,
  fillRatio: number,
  spillRatio: number,
  timestamp: number,
) {
  const mugWidth = headRadius * 0.84;
  const mugHeight = headRadius * 1.08;
  const anchor = {
    x: arm.hand.x,
    y: arm.hand.y - headRadius * 0.08,
  };
  const cupAngle = (-arm.wristAngle * Math.PI) / 180;
  const innerWidth = mugWidth * 0.64;
  const innerHeight = mugHeight * 0.7;
  const innerX = -innerWidth / 2;
  const innerY = -mugHeight * 0.42;
  const liquidHeight = innerHeight * clamp(fillRatio, 0, 1);

  context.save();
  context.translate(anchor.x, anchor.y);
  context.rotate(cupAngle);
  context.shadowColor = "rgba(15, 23, 42, 0.24)";
  context.shadowBlur = headRadius * 0.22;
  context.shadowOffsetY = headRadius * 0.08;
  context.fillStyle = "rgba(255, 248, 238, 0.16)";
  context.strokeStyle = "rgba(255, 244, 220, 0.78)";
  context.lineWidth = Math.max(2, headRadius * 0.05);
  drawRoundedRect(
    context,
    -mugWidth / 2,
    -mugHeight * 0.52,
    mugWidth,
    mugHeight * 0.92,
    mugWidth * 0.16,
  );
  context.fill();
  context.stroke();

  context.beginPath();
  context.ellipse(
    (side === "left" ? -1 : 1) * mugWidth * 0.52,
    -mugHeight * 0.06,
    mugWidth * 0.2,
    mugHeight * 0.18,
    0,
    0,
    Math.PI * 2,
  );
  context.stroke();

  context.save();
  drawRoundedRect(
    context,
    innerX,
    innerY,
    innerWidth,
    innerHeight,
    mugWidth * 0.12,
  );
  context.clip();
  context.fillStyle = "rgba(245, 158, 11, 0.9)";
  context.fillRect(
    innerX,
    innerY + innerHeight - liquidHeight,
    innerWidth,
    liquidHeight,
  );

  if (liquidHeight > 0) {
    context.fillStyle = "rgba(255, 248, 220, 0.85)";
    context.fillRect(
      innerX,
      innerY + innerHeight - liquidHeight - Math.max(2, headRadius * 0.05),
      innerWidth,
      Math.max(2, headRadius * 0.05),
    );
  }

  context.restore();
  context.restore();

  if (spillRatio <= 0.01 || fillRatio <= 0.01) {
    return;
  }

  const spillSide = normalizeAngle(arm.wristAngle - getCupSafeCenter(side)) < 0 ? -1 : 1;
  const rimLocalX = spillSide * mugWidth * 0.3;
  const rimLocalY = -mugHeight * 0.4;
  const rimX =
    anchor.x + rimLocalX * Math.cos(cupAngle) - rimLocalY * Math.sin(cupAngle);
  const rimY =
    anchor.y + rimLocalX * Math.sin(cupAngle) + rimLocalY * Math.cos(cupAngle);
  const streamLength = headRadius * (0.9 + spillRatio * 2.2);
  const streamDrift =
    spillSide * headRadius * (0.14 + spillRatio * 0.22) +
    Math.sin(timestamp / 150 + (side === "left" ? 0.4 : 1.1)) *
      headRadius *
      0.06;
  const streamGradient = context.createLinearGradient(
    rimX,
    rimY,
    rimX + streamDrift,
    rimY + streamLength,
  );

  streamGradient.addColorStop(0, "rgba(253, 224, 71, 0.96)");
  streamGradient.addColorStop(0.4, "rgba(251, 191, 36, 0.72)");
  streamGradient.addColorStop(1, "rgba(245, 158, 11, 0)");

  context.save();
  context.strokeStyle = streamGradient;
  context.lineCap = "round";
  context.lineWidth = Math.max(2, headRadius * (0.045 + spillRatio * 0.08));
  context.beginPath();
  context.moveTo(rimX, rimY);
  context.quadraticCurveTo(
    rimX + streamDrift * 0.7,
    rimY + streamLength * 0.36,
    rimX + streamDrift,
    rimY + streamLength,
  );
  context.stroke();

  for (let dropIndex = 0; dropIndex < 3; dropIndex += 1) {
    const progress =
      ((timestamp / 460 + dropIndex * 0.23 + (side === "left" ? 0.2 : 0)) % 1 + 1) % 1;
    const dropY = rimY + progress * streamLength;
    const dropX =
      rimX +
      streamDrift * Math.sin(progress * Math.PI) +
      spillSide * progress * headRadius * 0.04;

    drawPoint(
      context,
      {
        x: dropX,
        y: dropY,
      },
      "rgba(253, 224, 71, 0.82)",
      headRadius * (0.045 + (1 - progress) * 0.03),
    );
  }

  context.restore();
}

function buildArmPose(
  side: "left" | "right",
  shoulder: Point,
  fallbackHand: Point,
  handState: HandOverlayState | null,
  width: number,
  height: number,
  headRadius: number,
) {
  const wristPoint = getMirroredLandmarkPoint(handState?.landmarks[0], width, height);
  const palmPoint = getMirroredLandmarkPoint(handState?.landmarks[9], width, height);
  const hand = wristPoint && palmPoint
    ? {
        x: wristPoint.x * 0.72 + palmPoint.x * 0.28,
        y: wristPoint.y * 0.72 + palmPoint.y * 0.28,
      }
    : fallbackHand;
  const dx = hand.x - shoulder.x;
  const dy = hand.y - shoulder.y;
  const distance = Math.hypot(dx, dy) || 1;
  const normalX = -dy / distance;
  const normalY = dx / distance;
  const bendDirection = side === "left" ? -1 : 1;
  const bendAmount = Math.min(headRadius * 0.8, distance * 0.22) * bendDirection;
  const elbow = {
    x: shoulder.x + dx * 0.48 + normalX * bendAmount,
    y: shoulder.y + dy * 0.46 + normalY * bendAmount - headRadius * 0.08,
  };

  return {
    elbow,
    hand: {
      x: clamp(hand.x, headRadius, width - headRadius),
      y: clamp(hand.y, headRadius * 1.2, height - headRadius * 0.7),
    },
    handRadius: headRadius * (handState?.isFist ? 0.23 : 0.28),
    isFist: handState?.isFist ?? false,
    shoulder,
    wristAngle: handState?.wristAngle ?? (side === "left" ? -34 : 34),
  };
}

function buildPlayerPose(
  faceLandmarks: NormalizedLandmark[] | null,
  handStates: HandOverlayState[],
  width: number,
  height: number,
) {
  if (!faceLandmarks) {
    return null;
  }

  const bounds = getMirroredFaceBounds(faceLandmarks, width, height);

  if (!bounds) {
    return null;
  }

  const headRadius = clamp(
    Math.max(bounds.width, bounds.height) * 0.58,
    34,
    Math.min(width, height) * 0.12,
  );
  const headCenter = {
    x: bounds.centerX,
    y: bounds.centerY - headRadius * 0.06,
  };
  const shoulderY = headCenter.y + headRadius * 1.42;
  const shoulderWidth = Math.max(headRadius * 2.35, bounds.width * 1.48);
  const torsoHeight = headRadius * 2.75;
  const torsoWidth = shoulderWidth * 0.9;
  const shoulderLeft = {
    x: headCenter.x - shoulderWidth / 2,
    y: shoulderY,
  };
  const shoulderRight = {
    x: headCenter.x + shoulderWidth / 2,
    y: shoulderY,
  };
  const hips = {
    left: {
      x: headCenter.x - torsoWidth * 0.28,
      y: shoulderY + torsoHeight,
    },
    right: {
      x: headCenter.x + torsoWidth * 0.28,
      y: shoulderY + torsoHeight,
    },
  };
  const torsoCenter = {
    x: headCenter.x,
    y: shoulderY + torsoHeight * 0.55,
  };
  const neck = {
    height: headRadius * 0.45,
    width: headRadius * 0.56,
    x: headCenter.x,
    y: headCenter.y + headRadius * 0.95,
  };

  let leftHandState: HandOverlayState | null = null;
  let rightHandState: HandOverlayState | null = null;
  const remainingHands: HandOverlayState[] = [];

  for (const handState of handStates) {
    if (handState.handednessLabel === "왼손" && !leftHandState) {
      leftHandState = handState;
      continue;
    }

    if (handState.handednessLabel === "오른손" && !rightHandState) {
      rightHandState = handState;
      continue;
    }

    remainingHands.push(handState);
  }

  for (const handState of remainingHands) {
    const wristPoint = getMirroredLandmarkPoint(handState.landmarks[0], width, height);

    if (!wristPoint) {
      continue;
    }

    if (wristPoint.x < headCenter.x && !leftHandState) {
      leftHandState = handState;
      continue;
    }

    if (!rightHandState) {
      rightHandState = handState;
    }
  }

  const leftArm = buildArmPose(
    "left",
    shoulderLeft,
    {
      x: shoulderLeft.x - shoulderWidth * 0.18,
      y: shoulderY + torsoHeight * 0.38,
    },
    leftHandState,
    width,
    height,
    headRadius,
  );
  const rightArm = buildArmPose(
    "right",
    shoulderRight,
    {
      x: shoulderRight.x + shoulderWidth * 0.18,
      y: shoulderY + torsoHeight * 0.38,
    },
    rightHandState,
    width,
    height,
    headRadius,
  );

  return {
    arms: {
      left: leftArm,
      right: rightArm,
    },
    head: {
      radius: headRadius,
      x: headCenter.x,
      y: headCenter.y,
    },
    hitZones: [
      {
        radius: headRadius * 0.72,
        x: headCenter.x,
        y: headCenter.y,
      },
      {
        radius: torsoWidth * 0.34,
        x: torsoCenter.x,
        y: torsoCenter.y - torsoHeight * 0.18,
      },
      {
        radius: torsoWidth * 0.3,
        x: torsoCenter.x,
        y: torsoCenter.y + torsoHeight * 0.18,
      },
      {
        radius: leftArm.handRadius * 0.92,
        x: leftArm.hand.x,
        y: leftArm.hand.y,
      },
      {
        radius: rightArm.handRadius * 0.92,
        x: rightArm.hand.x,
        y: rightArm.hand.y,
      },
    ],
    hips,
    neck,
    shoulders: {
      left: shoulderLeft,
      right: shoulderRight,
    },
    torso: {
      center: torsoCenter,
      height: torsoHeight,
      width: torsoWidth,
    },
  };
}

function drawPlayerOverlay(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  playerPose: PlayerPose | null,
  playerSprites: PlayerSpriteMap,
  isHit: boolean,
  gameMode: GameMode,
  cupFill: CupFillState,
  showCupSpillEffect: boolean,
  timestamp: number,
  debugOverlay: DebugOverlayState | null,
  debugVisibility: DebugVisibility | null,
) {
  context.clearRect(0, 0, width, height);

  if (!playerPose) {
    if (debugVisibility) {
      drawDebugOverlay(context, width, height, debugOverlay, debugVisibility);
    }

    return;
  }

  const jacketColor = isHit ? "#7f1d1d" : "#29405c";
  const jacketTrim = isHit ? "#fca5a5" : "#d0b089";
  const sleeveColor = isHit ? "#991b1b" : "#37597d";
  const skinColor = "#cc9a74";
  const hairColor = "#3a2114";
  const head = playerPose.head;
  const torso = playerPose.torso;
  const neck = playerPose.neck;
  const shoulderSpan = playerPose.shoulders.right.x - playerPose.shoulders.left.x;
  const averageHipY = (playerPose.hips.left.y + playerPose.hips.right.y) / 2;

  context.save();
  context.globalCompositeOperation = "source-over";
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  if (isHit) {
    context.shadowColor = "rgba(248, 113, 113, 0.65)";
    context.shadowBlur = 22;
  }

  context.save();
  context.globalAlpha = PLAYER_ARM_ALPHA;
  for (const [arm, sprite, anchors] of [
    [
      playerPose.arms.left,
      playerSprites.leftArm,
      LEFT_ARM_SPRITE_SEGMENT_ANCHORS,
    ],
    [
      playerPose.arms.right,
      playerSprites.rightArm,
      RIGHT_ARM_SPRITE_SEGMENT_ANCHORS,
    ],
  ] as const) {
    const didDrawUpperArm = drawSegmentSprite(
      context,
      sprite,
      arm.shoulder,
      arm.elbow,
      anchors,
    );

    if (didDrawUpperArm) {
      continue;
    }

    drawLine(
      context,
      arm.shoulder,
      arm.elbow,
      sleeveColor,
      head.radius * 0.54,
    );
    drawLine(
      context,
      arm.shoulder,
      arm.elbow,
      jacketTrim,
      head.radius * 0.11,
    );
    drawPoint(context, arm.shoulder, jacketTrim, head.radius * 0.11);
    drawPoint(context, arm.elbow, jacketTrim, head.radius * 0.1);
  }
  context.restore();

  context.save();
  context.globalAlpha = PLAYER_ARM_ALPHA;
  for (const [arm, sprite] of [
    [playerPose.arms.left, playerSprites.leftHand],
    [playerPose.arms.right, playerSprites.rightHand],
  ] as const) {
    if (sprite) {
      continue;
    }

    drawLine(
      context,
      arm.elbow,
      arm.hand,
      sleeveColor,
      head.radius * 0.5,
    );
    drawLine(
      context,
      arm.elbow,
      arm.hand,
      jacketTrim,
      head.radius * 0.11,
    );
  }
  context.restore();

  context.save();
  context.globalAlpha = PLAYER_TORSO_ALPHA;
  const didDrawBody = drawSegmentSprite(
    context,
    playerSprites.body,
    {
      x: torso.center.x,
      y: playerPose.shoulders.left.y - head.radius * 0.14,
    },
    {
      x: torso.center.x,
      y: averageHipY,
    },
    BODY_SPRITE_SEGMENT_ANCHORS,
  );

  if (!didDrawBody) {
    context.fillStyle = jacketColor;
    context.strokeStyle = jacketTrim;
    context.lineWidth = Math.max(3, head.radius * 0.08);
    context.beginPath();
    context.moveTo(
      playerPose.shoulders.left.x - shoulderSpan * 0.08,
      playerPose.shoulders.left.y,
    );
    context.quadraticCurveTo(
      torso.center.x,
      torso.center.y - torso.height * 0.55,
      playerPose.shoulders.right.x + shoulderSpan * 0.08,
      playerPose.shoulders.right.y,
    );
    context.lineTo(playerPose.hips.right.x, playerPose.hips.right.y);
    context.quadraticCurveTo(
      torso.center.x,
      torso.center.y + torso.height * 0.12,
      playerPose.hips.left.x,
      playerPose.hips.left.y,
    );
    context.closePath();
    context.fill();
    context.stroke();

    context.fillStyle = "rgba(255, 247, 220, 0.18)";
    context.beginPath();
    context.moveTo(torso.center.x, playerPose.shoulders.left.y + head.radius * 0.18);
    context.lineTo(
      torso.center.x + head.radius * 0.16,
      playerPose.hips.right.y - head.radius * 0.42,
    );
    context.lineTo(
      torso.center.x - head.radius * 0.16,
      playerPose.hips.left.y - head.radius * 0.42,
    );
    context.closePath();
    context.fill();
  }

  context.fillStyle = skinColor;
  drawRoundedRect(
    context,
    neck.x - neck.width / 2,
    neck.y - neck.height / 2,
    neck.width,
    neck.height,
    neck.width * 0.36,
  );
  context.fill();
  context.restore();

  context.save();
  context.globalAlpha = PLAYER_HEAD_ALPHA;
  const didDrawHead = drawSegmentSprite(
    context,
    playerSprites.head,
    {
      x: head.x,
      y: head.y - head.radius * 1.52,
    },
    {
      x: head.x,
      y: head.y + head.radius * 1.08,
    },
    HEAD_SPRITE_SEGMENT_ANCHORS,
  );

  if (!didDrawHead) {
    context.fillStyle = hairColor;
    context.beginPath();
    context.ellipse(
      head.x,
      head.y,
      head.radius * 0.94,
      head.radius * 1.08,
      0,
      0,
      Math.PI * 2,
    );
    context.fill();

    context.fillStyle = "rgba(235, 209, 174, 0.18)";
    context.beginPath();
    context.ellipse(
      head.x,
      head.y - head.radius * 0.12,
      head.radius * 0.48,
      head.radius * 0.64,
      0,
      0,
      Math.PI * 2,
    );
    context.fill();

    context.strokeStyle = "rgba(255, 231, 204, 0.24)";
    context.lineWidth = head.radius * 0.08;
    context.beginPath();
    context.arc(
      head.x,
      head.y + head.radius * 0.08,
      head.radius * 0.44,
      Math.PI * 1.08,
      Math.PI * 1.92,
    );
    context.stroke();
  }
  context.restore();

  if (gameMode === "hard") {
    const leftSpillRatio = showCupSpillEffect
      ? getCupTiltSeverity("left", playerPose.arms.left.wristAngle)
      : 0;
    const rightSpillRatio = showCupSpillEffect
      ? getCupTiltSeverity("right", playerPose.arms.right.wristAngle)
      : 0;

    drawBeerCup(
      context,
      playerPose.arms.left,
      "left",
      head.radius,
      cupFill.left,
      leftSpillRatio,
      timestamp,
    );
    drawBeerCup(
      context,
      playerPose.arms.right,
      "right",
      head.radius,
      cupFill.right,
      rightSpillRatio,
      timestamp,
    );
  }

  context.save();
  context.globalAlpha = PLAYER_HAND_ALPHA;
  for (const [arm, sprite, anchors] of [
    [
      playerPose.arms.left,
      playerSprites.leftHand,
      LEFT_HAND_SPRITE_SEGMENT_ANCHORS,
    ],
    [
      playerPose.arms.right,
      playerSprites.rightHand,
      RIGHT_HAND_SPRITE_SEGMENT_ANCHORS,
    ],
  ] as const) {
    const didDrawLowerArm = drawSegmentSprite(
      context,
      sprite,
      arm.elbow,
      arm.hand,
      anchors,
    );

    if (didDrawLowerArm) {
      continue;
    }

    context.save();
    context.translate(arm.hand.x, arm.hand.y);
    context.rotate((-arm.wristAngle * Math.PI) / 180);
    context.fillStyle = skinColor;
    context.beginPath();
    context.ellipse(
      0,
      0,
      arm.handRadius * (arm.isFist ? 0.94 : 1.1),
      arm.handRadius * (arm.isFist ? 0.94 : 0.82),
      0,
      0,
      Math.PI * 2,
    );
    context.fill();
    context.strokeStyle = "rgba(67, 34, 18, 0.32)";
    context.lineWidth = 1.5;
    context.stroke();
    context.restore();
  }
  context.restore();

  context.restore();

  context.save();
  context.globalCompositeOperation = "screen";
  context.fillStyle = isHit ? "rgba(248, 113, 113, 0.08)" : "rgba(255, 234, 193, 0.03)";
  context.fillRect(0, 0, width, height);
  context.restore();

  if (debugVisibility) {
    drawDebugOverlay(context, width, height, debugOverlay, debugVisibility);
  }
}

function drawMotionScene(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  objects: SceneObject[],
  isActive: boolean,
  beerSprite: HTMLImageElement | null,
) {
  const centerX = width / 2;
  const centerY = height / 2;
  const maxRadius = Math.max(width, height) * 0.82;
  const overlay = context.createRadialGradient(
    centerX,
    centerY,
    0,
    centerX,
    centerY,
    maxRadius,
  );

  overlay.addColorStop(0, isActive ? "rgba(255, 214, 126, 0.14)" : "rgba(255, 214, 126, 0.08)");
  overlay.addColorStop(0.38, "rgba(22, 8, 6, 0.14)");
  overlay.addColorStop(1, "rgba(2, 6, 23, 0.74)");

  context.clearRect(0, 0, width, height);
  context.fillStyle = overlay;
  context.fillRect(0, 0, width, height);

  context.save();
  context.globalCompositeOperation = "screen";

  for (let index = 0; index < 10; index += 1) {
    const angle = (index / 10) * Math.PI - Math.PI / 2;
    const lineLength = Math.max(width, height) * 0.75;

    context.beginPath();
    context.moveTo(centerX, centerY);
    context.lineTo(
      centerX + Math.cos(angle) * lineLength,
      centerY + Math.sin(angle) * lineLength,
    );
    context.strokeStyle = `rgba(255, 232, 175, ${isActive ? 0.14 : 0.08})`;
    context.lineWidth = 1;
    context.stroke();
  }

  for (let ringIndex = 1; ringIndex <= 4; ringIndex += 1) {
    const progress = ringIndex / 4;

    context.beginPath();
    context.ellipse(
      centerX,
      centerY,
      width * 0.11 * progress,
      height * 0.095 * progress,
      0,
      0,
      Math.PI * 2,
    );
    context.strokeStyle = `rgba(255, 240, 202, ${0.1 - progress * 0.014})`;
    context.lineWidth = 1;
    context.stroke();
  }

  context.restore();

  const sortedObjects = [...objects].sort((left, right) => left.depth - right.depth);

  for (const object of sortedObjects) {
    drawSceneObject(
      context,
      getSceneObjectProjection(object, width, height),
      object.rotation,
      beerSprite,
    );
  }

  const centerGlow = context.createRadialGradient(
    centerX,
    centerY,
    0,
    centerX,
    centerY,
    Math.min(width, height) * 0.15,
  );

  centerGlow.addColorStop(0, isActive ? "rgba(255, 209, 115, 0.42)" : "rgba(255, 209, 115, 0.24)");
  centerGlow.addColorStop(0.5, "rgba(244, 114, 54, 0.12)");
  centerGlow.addColorStop(1, "rgba(14, 165, 233, 0)");
  context.fillStyle = centerGlow;
  context.beginPath();
  context.arc(centerX, centerY, Math.min(width, height) * 0.15, 0, Math.PI * 2);
  context.fill();

  context.fillStyle = "rgba(255, 246, 219, 0.95)";
  context.beginPath();
  context.arc(centerX, centerY, 3, 0, Math.PI * 2);
  context.fill();
}

function doesProjectionHitPlayer(
  projection: SceneObjectProjection,
  playerPose: PlayerPose | null,
) {
  if (!playerPose) {
    return false;
  }

  return playerPose.hitZones.some((zone) => {
    const dx = projection.impactX - zone.x;
    const dy = projection.impactY - zone.y;

    return Math.hypot(dx, dy) <= projection.impactRadius + zone.radius;
  });
}

export default function Home() {
  const stageRef = useRef<HTMLElement>(null);
  const backdropVideoRef = useRef<HTMLVideoElement>(null);
  const sceneCanvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const sceneContextRef = useRef<CanvasRenderingContext2D | null>(null);
  const analysisContextRef = useRef<CanvasRenderingContext2D | null>(null);
  const overlayContextRef = useRef<CanvasRenderingContext2D | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analysisFrameRequestRef = useRef<number | null>(null);
  const sceneFrameRequestRef = useRef<number | null>(null);
  const previousFrameRef = useRef<Uint8Array | null>(null);
  const lastAnalysisRef = useRef(0);
  const lastSignalAtRef = useRef<number | null>(null);
  const lastPlayerSeenAtRef = useRef(0);
  const lastSceneFrameRef = useRef(0);
  const motionDetectedRef = useRef(false);
  const analysisActiveRef = useRef(false);
  const sensitivityRef = useRef(4);
  const startCameraRef = useRef<() => Promise<void>>(async () => undefined);
  const faceLandmarkerRef = useRef<FaceLandmarkerInstance | null>(null);
  const faceLoadPromiseRef = useRef<Promise<FaceLandmarkerInstance> | null>(null);
  const handLandmarkerRef = useRef<HandLandmarkerInstance | null>(null);
  const handLoadPromiseRef = useRef<Promise<HandLandmarkerInstance> | null>(null);
  const faceModuleRef = useRef<VisionModule | null>(null);
  const lastFaceVideoTimeRef = useRef(-1);
  const changedPixelRatioRef = useRef(0);
  const beerSpriteRef = useRef<HTMLImageElement | null>(null);
  const playerSpritesRef = useRef<PlayerSpriteMap>(createEmptyPlayerSprites());
  const playerPoseRef = useRef<PlayerPose | null>(null);
  const playerHpRef = useRef(PLAYER_MAX_HP);
  const playerDamageCooldownUntilRef = useRef(0);
  const playerHitFlashUntilRef = useRef(0);
  const activeElapsedMsRef = useRef(0);
  const gameModeRef = useRef<GameMode>("normal");
  const beerDensityRef = useRef(DEFAULT_BEER_DENSITY);
  const cupFillRef = useRef<CupFillState>(createFullCupFill());
  const publishedCupFillRef = useRef<CupFillState>(createFullCupFill());
  const gameWonRef = useRef(false);
  const gameOverRef = useRef(false);
  const debugOverlayRef = useRef<DebugOverlayState>({
    faceLandmarks: null,
    handStates: [],
  });
  const renderSceneFrameRef = useRef<(timestamp: number) => void>(() => undefined);
  const sceneObjectsRef = useRef<SceneObject[]>(
    createSceneObjects(getSceneObjectCount(DEFAULT_BEER_DENSITY), "staggered"),
  );

  const [gameMode, setGameMode] = useState<GameMode>("normal");
  const [beerDensity, setBeerDensity] = useState(DEFAULT_BEER_DENSITY);
  const [goalDistanceMeters, setGoalDistanceMeters] = useState(
    DEFAULT_GOAL_DISTANCE_METERS,
  );
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [activeElapsedMs, setActiveElapsedMs] = useState(0);
  const [gameWon, setGameWon] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const [cupFill, setCupFill] = useState<CupFillState>(createFullCupFill);
  const [hardModeResult, setHardModeResult] = useState<HardModeResult | null>(null);
  const [playerHp, setPlayerHp] = useState(PLAYER_MAX_HP);
  const [sensitivity, setSensitivity] = useState(4);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isControlPanelOpen, setIsControlPanelOpen] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [debugVisibility, setDebugVisibility] = useState<DebugVisibility>({
    eyes: true,
    handJoints: true,
    mouth: true,
    nose: true,
  });

  const prepareSceneContext = () => {
    const canvas = sceneCanvasRef.current;
    const stage = stageRef.current;

    if (!canvas || !stage) {
      return null;
    }

    if (canvas.width !== stage.clientWidth || canvas.height !== stage.clientHeight) {
      canvas.width = stage.clientWidth;
      canvas.height = stage.clientHeight;
      sceneContextRef.current = canvas.getContext("2d", {
        alpha: true,
        desynchronized: true,
      });
    }

    const context =
      sceneContextRef.current ??
      canvas.getContext("2d", {
        alpha: true,
        desynchronized: true,
      });

    if (!context) {
      return null;
    }

    sceneContextRef.current = context;

    return {
      context,
      height: canvas.height,
      width: canvas.width,
    };
  };

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
    }

    const context = overlayContextRef.current ?? canvas.getContext("2d");

    if (!context) {
      return null;
    }

    overlayContextRef.current = context;

    return {
      context,
      height: canvas.height,
      width: canvas.width,
    };
  };

  const clearOverlay = () => {
    const canvas = overlayCanvasRef.current;
    const context = overlayContextRef.current ?? canvas?.getContext("2d");

    if (!canvas || !context) {
      return;
    }

    overlayContextRef.current = context;
    context.clearRect(0, 0, canvas.width, canvas.height);
  };

  const resetPlayerState = () => {
    const fullCupFill = createFullCupFill();

    playerPoseRef.current = null;
    debugOverlayRef.current = {
      faceLandmarks: null,
      handStates: [],
    };
    playerHpRef.current = PLAYER_MAX_HP;
    playerDamageCooldownUntilRef.current = 0;
    playerHitFlashUntilRef.current = 0;
    activeElapsedMsRef.current = 0;
    gameWonRef.current = false;
    gameOverRef.current = false;
    cupFillRef.current = fullCupFill;
    publishedCupFillRef.current = fullCupFill;
    lastPlayerSeenAtRef.current = 0;
    setPlayerHp(PLAYER_MAX_HP);
    setActiveElapsedMs(0);
    setGameWon(false);
    setGameOver(false);
    setCupFill(fullCupFill);
    setHardModeResult(null);
  };

  const syncBackdropPlayback = (shouldPlay: boolean) => {
    const backdropVideo = backdropVideoRef.current;

    if (!backdropVideo) {
      return;
    }

    if (shouldPlay) {
      if (backdropVideo.paused) {
        void backdropVideo.play().catch(() => undefined);
      }

      return;
    }

    if (!backdropVideo.paused) {
      backdropVideo.pause();
    }
  };

  const publishCupFillState = (nextCupFill: CupFillState) => {
    cupFillRef.current = nextCupFill;

    const publishedCupFill = publishedCupFillRef.current;
    const shouldPublish =
      Math.abs(nextCupFill.left - publishedCupFill.left) >= 0.01 ||
      Math.abs(nextCupFill.right - publishedCupFill.right) >= 0.01 ||
      (nextCupFill.left === 0 && publishedCupFill.left !== 0) ||
      (nextCupFill.right === 0 && publishedCupFill.right !== 0);

    if (!shouldPublish) {
      return;
    }

    const snapshot = {
      left: nextCupFill.left,
      right: nextCupFill.right,
    };

    publishedCupFillRef.current = snapshot;
    setCupFill(snapshot);
  };

  const resetGameRound = () => {
    previousFrameRef.current = null;
    lastSignalAtRef.current = null;
    lastSceneFrameRef.current = 0;
    motionDetectedRef.current = false;
    changedPixelRatioRef.current = 0;
    syncBackdropPlayback(false);
    sceneObjectsRef.current = createSceneObjects(
      getSceneObjectCount(beerDensityRef.current),
      "staggered",
    );
    resetPlayerState();
  };

  const handleModeChange = (nextMode: GameMode) => {
    if (nextMode === gameModeRef.current) {
      return;
    }

    gameModeRef.current = nextMode;
    setGameMode(nextMode);
    resetGameRound();
  };

  const handleControlPanelToggle = () => {
    setIsControlPanelOpen((current) => !current);
  };

  const handleDebugModeToggle = () => {
    setDebugMode((current) => !current);
  };

  const toggleDebugVisibility = (key: keyof DebugVisibility) => {
    setDebugVisibility((current) => ({
      ...current,
      [key]: !current[key],
    }));
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
            delegate: "CPU",
            modelAssetPath: FACE_LANDMARKER_MODEL_PATH,
          },
          minFaceDetectionConfidence: 0.6,
          minFacePresenceConfidence: 0.6,
          minTrackingConfidence: 0.5,
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
          runningMode: "VIDEO",
        },
      );

      faceModuleRef.current = vision;
      faceLandmarkerRef.current = faceLandmarker;

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
            delegate: "CPU",
            modelAssetPath: HAND_LANDMARKER_MODEL_PATH,
          },
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.5,
          numHands: 2,
          runningMode: "VIDEO",
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

    analysisFrameRequestRef.current = requestAnimationFrame(analyzeFrame);
  };

  const stopCamera = (nextState: CameraState = "idle") => {
    analysisActiveRef.current = false;

    if (analysisFrameRequestRef.current !== null) {
      cancelAnimationFrame(analysisFrameRequestRef.current);
      analysisFrameRequestRef.current = null;
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
    lastSceneFrameRef.current = 0;
    motionDetectedRef.current = false;
    changedPixelRatioRef.current = 0;
    syncBackdropPlayback(false);
    sceneObjectsRef.current = createSceneObjects(
      getSceneObjectCount(beerDensityRef.current),
      "staggered",
    );
    debugOverlayRef.current = {
      faceLandmarks: null,
      handStates: [],
    };

    clearOverlay();
    resetPlayerState();

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
      changedPixelRatioRef.current = 0;
      previousFrameRef.current = currentFrame;
      queueNextAnalysis();
      return;
    }

    const intensityScore = (totalDelta / pixelCount / 255) * 100;
    const changedRatio = (changedPixels / pixelCount) * 100;
    const combinedScore = Math.min(100, intensityScore * 0.6 + changedRatio * 1.8);
    const roundedRatio = Number(changedRatio.toFixed(1));
    const now = Date.now();

    changedPixelRatioRef.current = roundedRatio;

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
      const nextFaceLandmarks = faceLandmarksList[0] ?? null;

      debugOverlayRef.current = {
        faceLandmarks: nextFaceLandmarks,
        handStates,
      };

      personVisible = faceLandmarksList.length > 0 || handStates.length > 0;

      const stage = stageRef.current;
      const nextPlayerPose =
        stage && nextFaceLandmarks
          ? buildPlayerPose(
              nextFaceLandmarks,
              handStates,
              stage.clientWidth,
              stage.clientHeight,
            )
          : null;

      if (nextPlayerPose) {
        playerPoseRef.current = nextPlayerPose;
        lastPlayerSeenAtRef.current = now;
      } else if (now - lastPlayerSeenAtRef.current > PLAYER_POSE_HOLD_MS) {
        playerPoseRef.current = null;
      }
    } else if (now - lastPlayerSeenAtRef.current > PLAYER_POSE_HOLD_MS) {
      playerPoseRef.current = null;
    }

    if (combinedScore >= sensitivityRef.current && personVisible) {
      lastSignalAtRef.current = now;
    }

    const isActive =
      lastSignalAtRef.current !== null &&
      now - lastSignalAtRef.current < MOTION_HOLD_MS;

    if (isActive !== motionDetectedRef.current) {
      motionDetectedRef.current = isActive;
      syncBackdropPlayback(isActive && !gameWonRef.current && !gameOverRef.current);
    }

    previousFrameRef.current = currentFrame;
    queueNextAnalysis();
  };

  renderSceneFrameRef.current = (timestamp: number) => {
    if (
      lastSceneFrameRef.current !== 0 &&
      timestamp - lastSceneFrameRef.current < SCENE_FRAME_INTERVAL_MS
    ) {
      sceneFrameRequestRef.current = requestAnimationFrame(renderSceneFrameRef.current);
      return;
    }

    const scene = prepareSceneContext();
    const overlay = prepareOverlayContext();

    if (scene) {
      const { context, width, height } = scene;
      const deltaSeconds =
        lastSceneFrameRef.current === 0
          ? SCENE_FRAME_INTERVAL_MS / 1000
          : Math.min((timestamp - lastSceneFrameRef.current) / 1000, 0.08);
      const playerPose = playerPoseRef.current;
      const deltaMs = deltaSeconds * 1000;

      lastSceneFrameRef.current = timestamp;

      const previousElapsedMs = activeElapsedMsRef.current;
      let nextElapsedMs = previousElapsedMs;

      if (motionDetectedRef.current && !gameWonRef.current && !gameOverRef.current) {
        nextElapsedMs = clamp(
          previousElapsedMs + deltaMs,
          0,
          Math.max(goalDurationMs, previousElapsedMs),
        );
      }

      if (nextElapsedMs !== previousElapsedMs) {
        activeElapsedMsRef.current = nextElapsedMs;

        if (
          Math.floor(previousElapsedMs / 100) !== Math.floor(nextElapsedMs / 100) ||
          nextElapsedMs === goalDurationMs
        ) {
          setActiveElapsedMs(nextElapsedMs);
        }
      }

      if (
        !gameWonRef.current &&
        !gameOverRef.current &&
        nextElapsedMs >= goalDurationMs
      ) {
        gameWonRef.current = true;
        syncBackdropPlayback(false);
        setGameWon(true);

        if (gameModeRef.current === "hard") {
          setHardModeResult(
            getHardModeResult(
              cupFillRef.current,
              playerHpRef.current / PLAYER_MAX_HP,
            ),
          );
        }
      }

      const hasBeerPhaseStarted = nextElapsedMs >= BEER_START_DELAY_MS;
      const targetSceneObjectCount = getSceneObjectCount(beerDensityRef.current);
      let nextObjects = sceneObjectsRef.current;

      if (previousElapsedMs < BEER_START_DELAY_MS && hasBeerPhaseStarted) {
        nextObjects = createSceneObjects(targetSceneObjectCount, "staggered");
      }

      if (nextObjects.length > targetSceneObjectCount) {
        nextObjects = nextObjects.slice(0, targetSceneObjectCount);
      } else if (nextObjects.length < targetSceneObjectCount) {
        nextObjects = [
          ...nextObjects,
          ...Array.from(
            { length: targetSceneObjectCount - nextObjects.length },
            (_, index) => createSceneObject(nextObjects.length + index),
          ),
        ];
      }

      let roundFinished = gameWonRef.current || gameOverRef.current;

      if (
        gameModeRef.current === "hard" &&
        motionDetectedRef.current &&
        hasBeerPhaseStarted &&
        !roundFinished &&
        playerPose
      ) {
        const nextCupFill = {
          left: cupFillRef.current.left,
          right: cupFillRef.current.right,
        };

        nextCupFill.left = clamp(
          nextCupFill.left -
            deltaSeconds *
              CUP_SPILL_RATE *
              getCupTiltSeverity("left", playerPose.arms.left.wristAngle),
          0,
          1,
        );
        nextCupFill.right = clamp(
          nextCupFill.right -
            deltaSeconds *
              CUP_SPILL_RATE *
              getCupTiltSeverity("right", playerPose.arms.right.wristAngle),
          0,
          1,
        );

        publishCupFillState(nextCupFill);

        if (getCombinedCupFill(nextCupFill) <= 0.01) {
          gameOverRef.current = true;
          syncBackdropPlayback(false);
          setGameOver(true);
        }
      }

      roundFinished = gameWonRef.current || gameOverRef.current;
      const showCupSpillEffect =
        gameModeRef.current === "hard" &&
        motionDetectedRef.current &&
        hasBeerPhaseStarted &&
        !roundFinished;

      if (motionDetectedRef.current && hasBeerPhaseStarted && !roundFinished) {
        const sceneSpeed = getSceneSpeedFromPixelChange(changedPixelRatioRef.current);

        nextObjects = nextObjects.map((object) => {
          const nextDepth = object.depth + deltaSeconds * sceneSpeed;

          if (nextDepth > 1.18) {
            return createSceneObject(object.id);
          }

          return {
            ...object,
            depth: nextDepth,
            rotation: object.rotation + deltaSeconds * object.spin,
          };
        });
      }

      const now = performance.now();
      let tookDamage = false;

      nextObjects = nextObjects.map((object) => {
        if (
          !hasBeerPhaseStarted ||
          roundFinished ||
          !playerPose ||
          now < playerDamageCooldownUntilRef.current
        ) {
          return object;
        }

        const projection = getSceneObjectProjection(object, width, height);

        if (
          projection.spriteSize < playerPose.head.radius * 1.2 ||
          !doesProjectionHitPlayer(projection, playerPose)
        ) {
          return object;
        }

        tookDamage = true;
        playerDamageCooldownUntilRef.current = now + PLAYER_DAMAGE_COOLDOWN_MS;
        playerHitFlashUntilRef.current = now + PLAYER_HIT_FLASH_MS;
        const nextHp = Math.max(0, playerHpRef.current - BEER_DAMAGE);

        playerHpRef.current = nextHp;
        setPlayerHp(nextHp);

        return createSceneObject(object.id);
      });

      sceneObjectsRef.current = nextObjects;

      drawMotionScene(
        context,
        width,
        height,
        hasBeerPhaseStarted ? sceneObjectsRef.current : [],
        motionDetectedRef.current,
        beerSpriteRef.current,
      );

      if (overlay) {
        drawPlayerOverlay(
          overlay.context,
          overlay.width,
          overlay.height,
          playerPose,
          playerSpritesRef.current,
          tookDamage || now < playerHitFlashUntilRef.current,
          gameModeRef.current,
          cupFillRef.current,
          showCupSpillEffect,
          timestamp,
          debugOverlayRef.current,
          debugMode ? debugVisibility : null,
        );
      }
    } else if (overlay) {
      const showCupSpillEffect =
        gameModeRef.current === "hard" &&
        motionDetectedRef.current &&
        activeElapsedMsRef.current >= BEER_START_DELAY_MS &&
        !gameWonRef.current &&
        !gameOverRef.current;

      drawPlayerOverlay(
        overlay.context,
        overlay.width,
        overlay.height,
        playerPoseRef.current,
        playerSpritesRef.current,
        performance.now() < playerHitFlashUntilRef.current,
        gameModeRef.current,
        cupFillRef.current,
        showCupSpillEffect,
        timestamp,
        debugOverlayRef.current,
        debugMode ? debugVisibility : null,
      );
    }

    sceneFrameRequestRef.current = requestAnimationFrame(renderSceneFrameRef.current);
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
        audio: false,
        video: {
          facingMode: "user",
          height: { ideal: 720 },
          width: { ideal: 1280 },
        },
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
      lastSceneFrameRef.current = 0;
      motionDetectedRef.current = false;
      changedPixelRatioRef.current = 0;
      syncBackdropPlayback(false);
      analysisActiveRef.current = true;
      sceneObjectsRef.current = createSceneObjects(
        getSceneObjectCount(beerDensityRef.current),
        "staggered",
      );
      resetPlayerState();

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
    beerDensityRef.current = beerDensity;
  }, [beerDensity]);

  useEffect(() => {
    const beerSprite = new window.Image();
    const markReady = () => {
      beerSpriteRef.current = beerSprite;
    };

    beerSprite.decoding = "async";
    beerSprite.src = BEER_SPRITE_PATH;

    if (beerSprite.complete) {
      markReady();
    } else {
      beerSprite.addEventListener("load", markReady);
    }

    return () => {
      beerSprite.removeEventListener("load", markReady);

      if (beerSpriteRef.current === beerSprite) {
        beerSpriteRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const cleanupCallbacks = (
      Object.entries(PLAYER_SPRITE_PATHS) as Array<[PlayerSpriteKey, string]>
    ).map(([key, source]) => {
      const sprite = new window.Image();
      const markReady = () => {
        playerSpritesRef.current[key] = sprite;
      };

      sprite.decoding = "async";
      sprite.src = source;

      if (sprite.complete) {
        markReady();
      } else {
        sprite.addEventListener("load", markReady);
      }

      return () => {
        sprite.removeEventListener("load", markReady);

        if (playerSpritesRef.current[key] === sprite) {
          playerSpritesRef.current[key] = null;
        }
      };
    });

    return () => {
      cleanupCallbacks.forEach((cleanup) => cleanup());
    };
  }, []);

  useEffect(() => {
    sceneFrameRequestRef.current = requestAnimationFrame(renderSceneFrameRef.current);
    void startCameraRef.current();

    return () => {
      analysisActiveRef.current = false;

      if (analysisFrameRequestRef.current !== null) {
        cancelAnimationFrame(analysisFrameRequestRef.current);
      }

      if (sceneFrameRequestRef.current !== null) {
        cancelAnimationFrame(sceneFrameRequestRef.current);
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      faceLandmarkerRef.current?.close();
      handLandmarkerRef.current?.close();
    };
  }, []);

  const showPlaceholder = cameraState !== "ready";
  const currentPlayerPose = playerPoseRef.current;
  const leftCupAngle = currentPlayerPose?.arms.left.wristAngle ?? null;
  const rightCupAngle = currentPlayerPose?.arms.right.wristAngle ?? null;
  const leftCupUnsafeDelta =
    leftCupAngle === null ? null : getCupUnsafeDelta("left", leftCupAngle);
  const rightCupUnsafeDelta =
    rightCupAngle === null ? null : getCupUnsafeDelta("right", rightCupAngle);
  const isLeftCupSpilling = leftCupUnsafeDelta !== null && leftCupUnsafeDelta > 0;
  const isRightCupSpilling = rightCupUnsafeDelta !== null && rightCupUnsafeDelta > 0;
  const isBeerSpilling = isLeftCupSpilling || isRightCupSpilling;
  const fatigueRatio = Math.max(0, Math.min(100, playerHp)) / PLAYER_MAX_HP;
  const totalBeerRemainingPercent = Math.round(getCombinedCupFill(cupFill) * 100);
  const remainingFatiguePercent = Math.round(fatigueRatio * 100);
  const finalHardModeResult =
    hardModeResult ?? getHardModeResult(cupFill, playerHp / PLAYER_MAX_HP);
  const sceneObjectCount = getSceneObjectCount(beerDensity);
  const goalDurationMs = goalDistanceMeters * 1000;
  const distanceValue = formatDistanceMeters(activeElapsedMs);
  const enabledDebugLayerCount = Object.values(debugVisibility).filter(Boolean).length;
  const distanceProgressRatio = clamp(activeElapsedMs / Math.max(goalDurationMs, 1), 0, 1);
  const distanceProgressPercent = Math.round(distanceProgressRatio * 100);

  return (
    <main className={styles.page}>
      <section ref={stageRef} className={styles.stage}>
        <video
          ref={backdropVideoRef}
          className={styles.sceneBackdropVideo}
          src={BACKDROP_VIDEO_PATH}
          loop
          muted
          playsInline
          preload="auto"
          aria-hidden="true"
        />
        <div className={styles.sceneBackdrop} aria-hidden="true" />
        <canvas ref={sceneCanvasRef} className={styles.sceneCanvas} aria-hidden="true" />
        <video ref={videoRef} className={styles.cameraFeed} autoPlay muted playsInline />
        <canvas
          ref={overlayCanvasRef}
          className={styles.overlayCanvas}
          aria-hidden="true"
        />
        <div className={styles.hudTop}>
          <div
            className={`${styles.hpCard} ${
              playerHp <= 30 ? styles.hpCardDanger : ""
            }`}
          >
            <strong className={styles.sideGaugeValue}>{remainingFatiguePercent}%</strong>
            <div className={styles.hpBar} aria-hidden="true">
              <div
                className={styles.hpFill}
                style={{
                  transform: `scaleY(${fatigueRatio})`,
                }}
              />
            </div>
            <span className={styles.sideGaugeLabel}>피로도</span>
          </div>
          <div
            className={styles.miniMapCard}
            aria-label={`이동 거리 ${distanceValue}, 목표 ${goalDistanceMeters}m, 진행 ${distanceProgressPercent}%`}
          >
            <div className={styles.miniMapTrack} aria-hidden="true">
              <div
                className={styles.miniMapFill}
                style={{ transform: `scaleX(${distanceProgressRatio})` }}
              />
              <div
                className={styles.miniMapMarker}
                style={{ left: `${(distanceProgressRatio * 100).toFixed(1)}%` }}
              />
            </div>
            <strong className={styles.miniMapDistance}>{distanceValue}</strong>
          </div>
          {gameMode === "hard" && (
            <div
              className={`${styles.beerCard} ${
                isBeerSpilling || totalBeerRemainingPercent <= CUP_FILL_WARNING_RATIO * 100
                  ? styles.beerCardDanger
                  : ""
              }`}
            >
              <strong
                className={`${styles.sideGaugeValue} ${
                  isBeerSpilling ? styles.sideGaugeValueAlert : ""
                }`}
              >
                {totalBeerRemainingPercent}%
              </strong>
              <div className={styles.beerGaugeGroup}>
                <div className={styles.beerGauge}>
                  <div
                    className={`${styles.beerTrack} ${
                      isLeftCupSpilling ? styles.beerTrackDanger : ""
                    }`}
                    aria-hidden="true"
                  >
                    <div
                      className={styles.beerLiquid}
                      style={{ transform: `scaleY(${cupFill.left})` }}
                    />
                  </div>
                  <span className={styles.beerGaugeLabel}>L</span>
                </div>
                <div className={styles.beerGauge}>
                  <div
                    className={`${styles.beerTrack} ${
                      isRightCupSpilling ? styles.beerTrackDanger : ""
                    }`}
                    aria-hidden="true"
                  >
                    <div
                      className={styles.beerLiquid}
                      style={{ transform: `scaleY(${cupFill.right})` }}
                    />
                  </div>
                  <span className={styles.beerGaugeLabel}>R</span>
                </div>
              </div>
              <span className={styles.sideGaugeLabel}>맥주</span>
            </div>
          )}
        </div>
        {isControlPanelOpen && (
          <div id="game-control-panels" className={styles.hudBottom}>
            <div className={styles.modeSwitch} role="group" aria-label="게임 모드 선택">
              <button
                type="button"
                className={`${styles.modeButton} ${
                  gameMode === "normal" ? styles.modeButtonActive : ""
                }`}
                onClick={() => handleModeChange("normal")}
                aria-pressed={gameMode === "normal"}
              >
                Normal
              </button>
              <button
                type="button"
                className={`${styles.modeButton} ${
                  gameMode === "hard" ? styles.modeButtonActive : ""
                }`}
                onClick={() => handleModeChange("hard")}
                aria-pressed={gameMode === "hard"}
              >
                Hard
              </button>
            </div>
            <div className={`${styles.sliderCard} ${styles.debugCard}`}>
              <div className={styles.debugHeader}>
                <div>
                  <span className={styles.sliderLabel}>디버그 라벨</span>
                  <p className={styles.debugSummary}>
                    {debugMode
                      ? `${enabledDebugLayerCount}개 레이어 표시 중`
                      : "현재 오버레이 숨김"}
                  </p>
                </div>
                <button
                  type="button"
                  className={`${styles.debugModeButton} ${
                    debugMode ? styles.debugModeButtonActive : ""
                  }`}
                  onClick={handleDebugModeToggle}
                  aria-pressed={debugMode}
                >
                  {debugMode ? "Debug ON" : "Debug OFF"}
                </button>
              </div>
              <div className={styles.debugToggleGrid}>
                <button
                  type="button"
                  className={`${styles.debugToggleButton} ${
                    debugVisibility.eyes ? styles.debugToggleButtonActive : ""
                  }`}
                  onClick={() => toggleDebugVisibility("eyes")}
                  aria-pressed={debugVisibility.eyes}
                >
                  눈
                </button>
                <button
                  type="button"
                  className={`${styles.debugToggleButton} ${
                    debugVisibility.nose ? styles.debugToggleButtonActive : ""
                  }`}
                  onClick={() => toggleDebugVisibility("nose")}
                  aria-pressed={debugVisibility.nose}
                >
                  코
                </button>
                <button
                  type="button"
                  className={`${styles.debugToggleButton} ${
                    debugVisibility.mouth ? styles.debugToggleButtonActive : ""
                  }`}
                  onClick={() => toggleDebugVisibility("mouth")}
                  aria-pressed={debugVisibility.mouth}
                >
                  입
                </button>
                <button
                  type="button"
                  className={`${styles.debugToggleButton} ${
                    debugVisibility.handJoints ? styles.debugToggleButtonActive : ""
                  }`}
                  onClick={() => toggleDebugVisibility("handJoints")}
                  aria-pressed={debugVisibility.handJoints}
                >
                  손 관절
                </button>
              </div>
              <span className={styles.sliderHint}>
                얼굴 라벨은 눈·코·입 기준점으로, 손 라벨은 21개 관절명과 손목 각도로 표시됩니다
              </span>
            </div>
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
            <div className={styles.sliderCard}>
              <div className={styles.sliderHeader}>
                <span className={styles.sliderLabel}>맥주 등장량</span>
                <strong className={styles.sliderValue}>{beerDensity.toFixed(1)}x</strong>
              </div>
              <input
                className={styles.rangeInput}
                type="range"
                min={MIN_BEER_DENSITY.toString()}
                max={MAX_BEER_DENSITY.toString()}
                step="0.1"
                value={beerDensity}
                onChange={(event) => setBeerDensity(Number(event.target.value))}
                aria-label="맥주 등장량"
              />
              <span className={styles.sliderHint}>현재 동시 등장 {sceneObjectCount}잔</span>
            </div>
            <div className={styles.sliderCard}>
              <div className={styles.sliderHeader}>
                <span className={styles.sliderLabel}>게임 목표 거리</span>
                <strong className={styles.sliderValue}>{goalDistanceMeters}m</strong>
              </div>
              <input
                className={styles.rangeInput}
                type="range"
                min={MIN_GOAL_DISTANCE_METERS.toString()}
                max={MAX_GOAL_DISTANCE_METERS.toString()}
                step="1"
                value={goalDistanceMeters}
                onChange={(event) => setGoalDistanceMeters(Number(event.target.value))}
                aria-label="게임 목표 거리"
              />
              <span className={styles.sliderHint}>1초 움직이면 1m로 계산됩니다</span>
            </div>
          </div>
        )}
        <div className={styles.controlToggleDock}>
          <button
            type="button"
            className={`${styles.controlToggleButton} ${
              isControlPanelOpen ? styles.controlToggleButtonActive : ""
            }`}
            onClick={handleControlPanelToggle}
            aria-expanded={isControlPanelOpen}
            aria-controls="game-control-panels"
          >
            {isControlPanelOpen ? "컨트롤 패널 숨기기" : "컨트롤 패널 보기"}
          </button>
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
                `권한을 허용하면 움직인 시간 기준 5초 뒤부터 맥주가 등장하고, ${goalDistanceMeters}m를 이동하면 승리합니다.`}
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
        {(gameWon || gameOver) && (
          <div className={styles.resultOverlay}>
            <strong>{gameWon ? (gameMode === "hard" ? finalHardModeResult.grade : "승리") : "실패"}</strong>
            <button type="button" className={styles.retryButton} onClick={resetGameRound}>
              다시 시작
            </button>
          </div>
        )}
        <canvas ref={analysisCanvasRef} className={styles.hiddenCanvas} aria-hidden="true" />
      </section>
    </main>
  );
}
