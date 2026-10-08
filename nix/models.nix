{
  lib,
  fetchurl,
  runCommand,
  unzip,
}: let
  task = fetchurl {
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task?generation=1683136941916318";
    hash = "sha256-ZBhOIpsmMQe8K4BMZiXbE0H/K7cxh0sLzC/mVE4Lyf8=";
  };
  faceDetectorOnnx = fetchurl {
    url = "https://huggingface.co/fernandotonon/QtMeshEditor-blazeface-onnx/resolve/50f2c66ffbdf84beae8c267df2b49e5c5a5162e9/face_detector.onnx";
    hash = "sha256-AqBNXTfDVY3E1SdPf48PDwGslORsX/ss7oI5XUfiMYE=";
  };
  faceLandmarksOnnx = fetchurl {
    url = "https://huggingface.co/senty-au/face_landmarks_detector-ONNX/resolve/337d58218b5b1cc597ca3c67360880b920f6ce7b/onnx/model.onnx";
    hash = "sha256-fW6C3ugqHcpfvdsoKzzHRXGDOlMN4xf8Iq4yXDNYvus=";
  };
in
  runCommand "eye-mediapipe-models" {
    nativeBuildInputs = [unzip];
    meta.license = lib.licenses.asl20;
  } ''
    mkdir -p $out/mediapipe/tflite
    unzip -j ${task} face_detector.tflite face_landmarks_detector.tflite -d $out/mediapipe/tflite
    cp ${faceDetectorOnnx} $out/mediapipe/face_detector.onnx
    cp ${faceLandmarksOnnx} $out/mediapipe/face_landmarks_detector.onnx
  ''
