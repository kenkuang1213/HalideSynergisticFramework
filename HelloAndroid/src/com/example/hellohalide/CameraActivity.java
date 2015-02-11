package com.example.hellohalide;

import android.app.Activity;
import android.os.Bundle;
import android.hardware.Camera;
import android.util.Log;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.view.SurfaceView;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;
import android.widget.Toast;


public class CameraActivity extends Activity {
    public static int workload=100;
    private static final String TAG = "CameraActivity";
    public SeekBar seekBar;
    private Camera camera;
    private CameraPreview preview;
    private SurfaceView filtered;
	
    public static Camera getCameraInstance() {
        Camera c = null;
        try {
            c = Camera.open();
        } catch (Exception e) {
            Log.d(TAG, "Could not open camera");
        }
        return c;
    }

    @Override
    public void onCreate(Bundle b) {
        super.onCreate(b);

        setContentView(R.layout.main);

        // Create a canvas for drawing stuff on
        filtered = new SurfaceView(this);

        // Create our Preview view and set it as the content of our activity.
        preview = new CameraPreview(this, filtered);

        FrameLayout layout = (FrameLayout) findViewById(R.id.camera_preview);
		  LinearLayout  ll=(LinearLayout) findViewById(R.id.Linear);
               seekBar=(SeekBar) findViewById(R.id.seekBar);
               seekBar.setProgress(100);
    layout.addView(preview);

      layout.addView(filtered);
      seekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {

            @Override
            public void onProgressChanged(SeekBar seekBar,int progresValue,boolean fromUser){
                workload=progresValue;
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekbar){

            }
            @Override
            public void onStopTrackingTouch(SeekBar seekbar){

                // Toast.makeText(CameraActivity.this,"Worklad : "+workload,Toast.LENGTH_SHORT).show();
            }
      });
	
		
       filtered.setZOrderMediaOverlay(true);
	seekBar.bringToFront();
    }

    @Override
    public void onResume() {
        super.onResume();
        camera = getCameraInstance();
        preview.setCamera(camera);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (camera != null) {
            preview.setCamera(null);
            camera.release();
            camera = null;
        }
    }
}
