package org.tensorflow.demo;

import android.content.Context;
import android.graphics.Bitmap;

import com.google.android.gms.analytics.GoogleAnalytics;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.android.gms.location.LocationServices;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.io.ByteArrayOutputStream;

import android.graphics.RectF;
import android.location.Address;
import android.location.Geocoder;
import android.location.LocationManager;
import android.os.Bundle;
import android.util.Base64;

import java.lang.reflect.Array;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import android.location.LocationManager;
import android.location.Location;
import android.location.LocationListener;

import java.util.Locale;
import java.util.Map;

class FirebaseHandler {
    DatabaseReference mRootRef ;
    int lastBox = -1;
    Context mContext;
    ArrayList<String> currLocation;
    Location currLoc = null;
    Location prevLoc = null;
    LocationListener locListener = new LocationListener() {
        @Override
        public void onLocationChanged(Location location) {

        }

        @Override
        public void onStatusChanged(String s, int i, Bundle bundle) {

        }

        @Override
        public void onProviderEnabled(String s) {

        }

        @Override
        public void onProviderDisabled(String s) {

        }
    };

    public  FirebaseHandler(Context mContext){
        this.mContext = mContext;
        this.mRootRef = FirebaseDatabase.getInstance().getReference();
        this.currLocation = new ArrayList<String>();

    }
    public void addReport(Bitmap img, RectF pothole, String confidense){
        HashMap<String, String> report = new HashMap<>();
        report.put("img", encodeTobase64(img));
        ArrayList<String> location= currLocation;
        report.put("lat", location.get(0));
        report.put("long", location.get(1));
        report.put("street", getStreet(Double.parseDouble(location.get(0)), Double.parseDouble(location.get(1))));
        report.put("status", "Submitted");
        //report.put("pothole", pothole);
        report.put("pothole", String.format("%f,%f,%f,%f", pothole.left, pothole.top,pothole.width(), pothole.height()));
        report.put("confidense", confidense);
        report.put("timeStamp", getTime());
        //sending data
        String key = mRootRef.child("potholes").push().getKey();
        //String key = mRootRef.child("potholes").push().getKey();
        Map<String, Object> childUpdate= new HashMap<>();
        childUpdate.put("/potholes/" + key, report);
        this.mRootRef.updateChildren(childUpdate);

    }

    private String getStreet(double LATITUDE, double LONGITUDE) {
        String strAdd = "";
        Geocoder geocoder = new Geocoder(this.mContext, Locale.getDefault());
        try {
            List<Address> addresses = geocoder.getFromLocation(LATITUDE, LONGITUDE, 1);
            if (addresses != null) {
                strAdd = addresses.get(0).getAddressLine(0);
                //Log.w("My Current loction address", strReturnedAddress.toString());
            } else {
                //Log.w("My Current loction address", "No Address returned!");
            }
        } catch (Exception e) {
            e.printStackTrace();
            //Log.w("My Current loction address", "Canont get Address!");
        }
        return strAdd;
    }

    public Boolean validateLocation(){
        if (currLoc != null && prevLoc != null){
            if (currLoc.getLatitude() == prevLoc.getLatitude() && currLoc.getLongitude() == prevLoc.getLongitude()){
                return false;
            }
            return true;
        }
        return true;
    }

    public String getTime(){
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("dd-MM-yyyy-hh-mm-ss");
        String format = simpleDateFormat.format(new Date());
        return format;
    }

    public void fetchLocation(){
        ArrayList<String> latlong = new ArrayList<String>();
        LocationManager lm = (LocationManager) this.mContext.getSystemService(Context.LOCATION_SERVICE);
        lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 500.0f,locListener );
        /* Loop over the array backwards, and if you get an accurate location, then breakout the loop*/
        this.prevLoc = this.currLoc;
        this.currLoc = lm.getLastKnownLocation(LocationManager.GPS_PROVIDER);

        double[] gps = new double[2];
        if (this.currLoc != null) {
            latlong.add(Double.toString(this.currLoc.getLatitude()));
            latlong.add(Double.toString(this.currLoc.getLongitude()));
        }
        else{
            latlong.add("0");
            latlong.add("0");
        }

        currLocation = latlong;
    }

    public static String encodeTobase64(Bitmap image) {
        Bitmap immagex = image;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        immagex.compress(Bitmap.CompressFormat.JPEG, 100, baos);
        byte[] b = baos.toByteArray();
        String imageEncoded = Base64.encodeToString(b, Base64.DEFAULT);
        return imageEncoded;
    }
}

