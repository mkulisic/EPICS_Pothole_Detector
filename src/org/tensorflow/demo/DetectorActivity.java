/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;
import android.content.res.AssetFileDescriptor;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.channels.FileChannel;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.provider.ContactsContract;
import android.renderscript.RenderScript;
import android.util.Size;
import android.util.TypedValue;
import java.nio.ByteBuffer;

import android.view.Display;
import android.view.Surface;
import android.widget.Toast;
import java.io.IOException;
import java.text.FieldPosition;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import java.util.Map;
import java.util.Random;

import org.tensorflow.demo.R; // Explicit import needed for internal Google builds.

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged multibox model.
  private static final int MB_INPUT_SIZE = 224;
  private static final int MB_IMAGE_MEAN = 128;
  private static final float MB_IMAGE_STD = 128;
  private static final String MB_INPUT_NAME = "ResizeBilinear";
  private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
  private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
  private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String MB_LOCATION_FILE =
      "file:///android_asset/multibox_location_priors.txt";

  /**
   *constants for TFLITE
  **/
  private static final int BLOCK_SIZE = 32;
  private static final int IMG_WIDTH = 416;
  private static final int IMG_HEIGHT = 416;
  private static final int NUM_CLASS = 1;
  private static final int NUM_BOXES = 5;
  private static final int MAX_R = 10;
  private static final double THRESH = 0.6;
  private static final double OVERLAP_THRESH = 0.5;
  private static final int NUM_CHANNELS = 3;



  private static final double IMAGE_REDUCTION_FACTOR = 0.6;
  private static final int POSSIBLE_BOXES = 5;
  private Random ran = new Random();

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE =
      "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

  // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
  // must be manually placed in the assets/ directory by the user.
  // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
  // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
  // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
  //private static final String YOLO_MODEL_FILE = "file:///android_asset/yolov3-tiny-nhwc.pb";
  private static final String YOLO_MODEL_FILE = "file:///android_asset/yolov2-tiny-cropped.pb";
  private static final int YOLO_INPUT_SIZE = 416;

  //yolov2-tiny
  private static final String YOLO_INPUT_NAME = "input";
  private static final String YOLO_OUTPUT_NAMES = "output";

  //yolov3-tiny-nhwc
  //private static final String YOLO_INPUT_NAME = "inputs:0";
  //private static final String YOLO_OUTPUT_NAMES = "output_boxes:0";

  //ultimate_yolov3
  //private static final String YOLO_INPUT_NAME = "yolov3-tiny/net1";
  //private static final String YOLO_OUTPUT_NAMES = "yolov3-tiny/convolutional10/BiasAdd";
  private static final int YOLO_BLOCK_SIZE = 32;

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
  // or YOLO.
  private enum DetectorMode {
    TF_OD_API, MULTIBOX, YOLO;
  }
  private static final DetectorMode MODE = DetectorMode.YOLO;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
  private static final float MINIMUM_CONFIDENCE_YOLO = 0.7f;

  private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(1920, 1080);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;
  //int newWidth = 0;
  //int widthOffset = 0;
  int newHeight = 0;
  int heightOffset = 0;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  public FirebaseHandler ReportHandler = new FirebaseHandler(this);
  private BorderedText borderedText;
  //TFLITE Tests
  private boolean modalLoaded;
  private TFliteYoloDetector detectorTFLITE;
  private RenderScript rs;
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {

    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;
    if (MODE == DetectorMode.YOLO) {
      detector =
          TensorFlowYoloDetector.create(
              getAssets(),
              YOLO_MODEL_FILE,
              YOLO_INPUT_SIZE,
              YOLO_INPUT_NAME,
              YOLO_OUTPUT_NAMES,
              YOLO_BLOCK_SIZE);
      cropSize = YOLO_INPUT_SIZE;
      //TFLITE
        rs = RenderScript.create(getApplicationContext());
        loadModel(YOLO_MODEL_FILE);
    } else if (MODE == DetectorMode.MULTIBOX) {
      detector =
          TensorFlowMultiBoxDetector.create(
              getAssets(),
              MB_MODEL_FILE,
              MB_LOCATION_FILE,
              MB_IMAGE_MEAN,
              MB_IMAGE_STD,
              MB_INPUT_NAME,
              MB_OUTPUT_LOCATIONS_NAME,
              MB_OUTPUT_SCORES_NAME);
      cropSize = MB_INPUT_SIZE;
    } else {
      try {
        detector = TensorFlowObjectDetectionAPIModel.create(
            getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final IOException e) {
        LOGGER.e(e, "Exception initializing classifier!");
        Toast toast =
            Toast.makeText(
                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();
    //widthOffset = (int)(previewWidth*IMAGE_REDUCTION_FACTOR/2);
    //newWidth = previewWidth - widthOffset*2;
    heightOffset = (int)(previewHeight * IMAGE_REDUCTION_FACTOR / 2);
    newHeight = previewHeight - heightOffset * 2;


    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            newHeight, newHeight,
            cropSize, cropSize,
            0, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });
  }

  OverlayView trackingOverlay;

    protected void loadModel(final String modalPath) {
        new Thread(new Runnable() {
            public void run() {
                try {
                    String modalPathKey = "yolov2-tiny-cropped.tflite";
                    ByteBuffer modalData = loadModalFile(modalPathKey, getAssets());
                    rs = RenderScript.create(getApplicationContext());
                    detectorTFLITE = new TFliteYoloDetector(
                            rs,
                            modalData,
                            BLOCK_SIZE,
                            IMG_WIDTH,
                            IMG_HEIGHT,
                            NUM_CLASS,
                            NUM_BOXES,
                            MAX_R,
                            0.2,
                            OVERLAP_THRESH,
                            NUM_CHANNELS
                            );
                    modalLoaded=true;
                    //result.success("Modal Loaded Sucessfully");
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("Modal failed to loaded");
                }
            }
        }).start();
    }

    public ByteBuffer loadModalFile(String model, AssetManager assetManager) throws IOException {
        AssetFileDescriptor AFD = assetManager.openFd(model);
        FileInputStream inputStream = new FileInputStream(AFD.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        //long startOffset = fileDescriptor.getStartOffset();
        //long declaredLength = fileDescriptor.getDeclaredLength();
        long startOffset = AFD.getStartOffset();
        long declaredLength = AFD.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");



    //int innerOffset = new Random().nextInt(POSSIBLE_BOXES);
    int innerOffset = (int) ((ran.nextGaussian() * 0.5) + (POSSIBLE_BOXES/2));
    if (innerOffset < 0){
      innerOffset = 0;
    }
    if (innerOffset >= POSSIBLE_BOXES){
      innerOffset = POSSIBLE_BOXES - 1;
    }

    //final int heightOffset = innerOffset * (previewHeight - newWidth)/ POSSIBLE_BOXES;
    final int widthOffset = innerOffset * (previewWidth - newHeight) / POSSIBLE_BOXES;
    //final int heightOffset = (int)(previewHeight*reduceFactor/2);
    //int newHeight = previewHeight - heightOffset*2;

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Bitmap tmp = Bitmap.createBitmap(rgbFrameBitmap, widthOffset, heightOffset, newHeight, newHeight);
    //final Bitmap tmp = Bitmap.createBitmap(rgbFrameBitmap,0, 0, previewWidth, previewHeight);
    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];

    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(tmp, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }


    runInBackground(
            new Runnable() {
              @Override
              public void run() {
                if (!modalLoaded){
                  System.out.println("Model is not loaded");
                }
                try {
                  List<Map<String, Object>> prediction = detectorTFLITE.detect(croppedBitmap);
                  System.out.println("Prediction" + prediction.toString());

                  ReportHandler.fetchLocation();
                  if(ReportHandler.validateLocation()) {
                    for (int i = 0; i < prediction.size(); i++) {
                      if (prediction.get(i) != null) {
                        Map<String, Object> result = prediction.get(i);
                        //fetch location
                        Map<String, Float> rectF = (Map<String, Float>) result.get("rect");
                        RectF location = new RectF(rectF.get("left"), rectF.get("top"), rectF.get("right"), rectF.get("bottom"));
                        if (location != null && (float) result.get("confidence") >= MINIMUM_CONFIDENCE_YOLO) {
                          cropToFrameTransform.mapRect(location);
                          ReportHandler.addReport(tmp, location, result.get("confidence").toString());

                          location.offset(widthOffset, heightOffset);
                          //result.setLocation(location);
                          //mappedRecognitions.add(result);
                        }
                      }
                    }
                  }

                  computingDetection = false;
                } catch (Exception e){
                  e.printStackTrace();
                  computingDetection = false;
                }
              }
            });
    /*
    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
              case MULTIBOX:
                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                break;
              case YOLO:
                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();
            ReportHandler.fetchLocation();
            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();

              if (location != null && result.getConfidence() >= minimumConfidence) {

                canvas.drawRect(location, paint);
                cropToFrameTransform.mapRect(location);
                ReportHandler.addReport(tmp, location, result.getConfidence().toString());

                location.offset(widthOffset, heightOffset);
                result.setLocation(location);
                //mappedRecognitions.add(result);
              }
            }

            //tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            //trackingOverlay.postInvalidate();

            requestRender();
            computingDetection = false;
          }
        });*/
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }
}
