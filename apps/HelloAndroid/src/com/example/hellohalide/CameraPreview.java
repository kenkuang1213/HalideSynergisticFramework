package com.example.hellohalide;

import android.hardware.Camera;
import android.util.Log;
import java.io.IOException;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.Surface;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;
import android.widget.Toast;
import android.widget.TextView;

/** A basic Camera preview class */
public class CameraPreview extends SurfaceView
    implements SurfaceHolder.Callback, Camera.PreviewCallback {
    private static final String TAG = "CameraPreview";
    private TextView textView;
    private Camera mCamera;
    private SurfaceView mFiltered;
    private byte[] mPreviewData;
    private int[] returnData;
    // Link to native Halide code
    static {
        System.loadLibrary("native");
    }
    private static native void processFrame(byte[] src, int w, int h, int workload,int[] returnData,Surface dst);

    public CameraPreview(Context context, SurfaceView filtered) {
        super(context);
        mFiltered = filtered;
        mFiltered.getHolder().setFormat(ImageFormat.YV12);
        mPreviewData = null;
        returnData=new int[2];

        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        getHolder().addCallback(this);
    }

    public void onPreviewFrame(byte[] data, Camera camera) {
        if (camera != mCamera) {
            Log.d(TAG, "Unknown Camera!");
            return;
        }
        if (mFiltered.getHolder().getSurface().isValid()) {
            // SeekBar seekBar= (SeekBar) findViewById(R.id.seekBar);
            // CameraActivity.workload=seekBar.getProgress();
            // int workload=0;
              // Log.d(TAG, "Worklad : "+    CameraActivity.workload);
            // Toast.makeText(CameraPreview.this,"Worklad : "+workload,Toast.LENGTH_SHORT).show();

            Camera.Size s = camera.getParameters().getPreviewSize();
            if(!CameraActivity.autoChecked)
                processFrame(data, s.width, s.height, CameraActivity.workload,returnData,mFiltered.getHolder().getSurface());
            else{
                processFrame(data, s.width, s.height, -1,returnData,mFiltered.getHolder().getSurface());
                CameraActivity.seekBar.setProgress(returnData[1]);
            }
            CameraActivity.textView.setText(""+returnData[0]);
        } else {
            Log.d(TAG, "Invalid Surface!");
        }

        // re-enqueue this buffer
        camera.addCallbackBuffer(data);
    }

    private void startPreview(SurfaceHolder holder) {
        if (mCamera == null) {
            return;
        }
        try {
            configureCamera();
            mCamera.setPreviewCallbackWithBuffer(this);
            mCamera.setPreviewDisplay(holder);
            mCamera.startPreview();
        } catch (Exception e){
            Log.d(TAG, "Error starting camera preview: " + e.getMessage());
        }
    }

    private void stopPreview() {
        if (mCamera == null) {
            return;
        }
        try {
              mCamera.stopPreview();
        } catch (Exception e){
              // ignore: tried to stop a non-existent preview
              Log.d(TAG, "tried to stop a non-existent preview");
        }
    }
    private void configureCamera() {
        Camera.Parameters p = mCamera.getParameters();
        Camera.Size s = p.getPreviewSize();
        Log.d(TAG, "Camera Preview Size: " + s.width + "x" + s.height);
        p.setPreviewFormat(ImageFormat.YV12);
        if (mPreviewData == null) {
            int stride = ((s.width + 15) / 16) * 16;
            int y_size = stride * s.height;
            int c_stride = ((stride/2 + 15) / 16) * 16;
            int c_size = c_stride * s.height/2;
            int size = y_size + c_size * 2;
            mPreviewData = new byte[size];
        }
        mCamera.addCallbackBuffer(mPreviewData);
        mCamera.setParameters(p);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        Log.d(TAG, "surfaceCreated");
        startPreview(holder);
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.d(TAG, "surfaceDestroyed");
        stopPreview();
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        Log.d(TAG, "surfaceChanged");
        stopPreview();
        configureCamera();
        startPreview(holder);
    }

    public void setCamera(Camera c) {
        if (mCamera != null) {
            mCamera.stopPreview();
        }
        mCamera = c;
        if (mCamera != null) {
            startPreview(getHolder());
        }
    }
}
