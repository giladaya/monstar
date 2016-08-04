$(window).load(function() {
    "use strict";

    // screen.addEventListener('onorientationchange', function() {

    // });

    function onGameStart () {
      compatibility.requestFullScreen(document.documentElement);
      screen.orientation.lock('landscape').then(function() {
        console.log(screen.orientation);
      }, 
      function(error) {
        console.log(error);
        document.exitFullscreen()
      });
    }

    onGameStart();


    // lets do some fun
    var video = document.getElementById('webcam');
    var canvas = document.getElementById('canvas');

    function initVideo() {
      try {
          var attempts = 0;
          var readyListener = function(event) {
              findVideoSize();
          };
          var findVideoSize = function() {
              if (video.videoWidth > 0 && video.videoHeight > 0) {
                  video.removeEventListener('loadeddata', readyListener);
                  onDimensionsReady(video.videoWidth, video.videoHeight);
              } else {
                  if(attempts < 10) {
                      attempts++;
                      setTimeout(findVideoSize, 200);
                  } else {
                      onDimensionsReady(640, 480);
                  }
              }
          };
          var onDimensionsReady = function(width, height) {
            //start the app
            demo_app(width, height);
            compatibility.requestAnimationFrame(tick);
          };

          video.addEventListener('loadeddata', readyListener);

          compatibility.getUserMedia({
                video: {facingMode: {exact: 'environment'}}
            }, 
            onGumSuccess, 
            onGumError);
      } catch (error) {
          $('#canvas').hide();
          $('#log').hide();
          $('#no_rtc').html('<h4>Something went wrong...</h4>');
          $('#no_rtc').show();
      }
    }


    function onGumSuccess(stream){
      try {
          video.src = compatibility.URL.createObjectURL(stream);
      } catch (error) {
          video.src = stream;
      }
      setTimeout(function() {
          video.play();
      }, 500);
    }
    function onGumError(error){
      $('#canvas').hide();
      $('#log').hide();
      $('#no_rtc').html('<h4>WebRTC not available.</h4>');
      $('#no_rtc').show();
    }



    var stat = new profiler();

    var gui,options,ctx,canvasWidth,canvasHeight;
    var classifiers = [
        jsfeat.haar.frontalface,
        jsfeat.haar.profileface,
    ]
    // var classifier = jsfeat.haar.frontalface;

    var max_work_size = 160;

    //interest point variables
    var corners_img_u8, corners; 
    //tracking variables
    var curr_img_pyr, prev_img_pyr, point_count, point_status, prev_xy, curr_xy; 

    var FRAMES_BETWEEN_DETECTS = 50;
    var MAX_TTL = 15; //max frames for a face not to be detected to be considered gone
    var MAX_POINTS = 200;
    var DETECTS_INTERVAL = 1000; //time between detects in ms
    var MIN_FACES = 4;

    //types
    var TYPE_FLOWER = 0;
    var TYPE_CHERRY = 1;

    var lastDetectTime;
    var startTime = Date.now();
    var frames = 0;
    var score = 0;
    var faces = {};
    var point_attr; //point to face lookup

    var mx; //moster x location
    var mh; //monster height

    /** 
   * Constructor for a Face object
   * @param cx face center x
   * @param cy face center y
   */
    function Face(x, y, w, h, type, score){
        type = type || TYPE_FLOWER;
        score = score || 1;
        var coords = {
          'x': x,
          'y': y,
          'w': w,
          'h': h,
          'cx': x + w/2,
          'cy': y + h/2
        };
        return {
          'id': ~~(Math.random() * 10000000),
          'ttl': MAX_TTL,
          'coords': coords,
          'old_coords': coords,
          'is_stale': false,
          'is_live': true,
          'points': [], //related tracking points indexes
          'age': 0, //age since birth / since death
          'type': type, //object type
          'score': score //object score
        }
    }

    function setFaceCoords(face, x, y) {
      face.old_coords = face.coords;
      face.coords = Object.assign({}, face.coords, {
          x: x,
          y: y,
          cx: x + face.coords.w/2,
          cy: y + face.coords.h/2
      });
    }

    var demo_opt = function(){
        //detector
        this.min_scale = 2;
        this.scale_factor = 1.15;
        this.equalize_histogram = false;
        this.use_canny = true;
        this.edges_density = 0.13;

        //points
        this.lap_thres = 10;
        this.eigen_thres = 10;
        this.threshold = 20;
        this.show_track_pts = true;

        //tracker
        this.win_size = 20;
        this.max_iterations = 30;
        this.epsilon = 0.01;
        this.min_eigen = 0.001;
    }

    function demo_app(videoWidth, videoHeight) {
        canvasWidth  = canvas.width;
        // canvasHeight = canvas.height;
        canvasHeight = ~~(canvas.width * window.innerHeight / window.innerWidth);
        canvas.height = canvasHeight;
        ctx = canvas.getContext('2d');

        ctx.fillStyle = "rgb(0,255,0)";
        ctx.strokeStyle = "rgb(0,255,0)";

        var scale = Math.min(max_work_size/videoWidth, max_work_size/videoHeight);
        var workWidth = (videoWidth*scale)|0;
        var workHeight = (videoHeight*scale)|0;

        mx = canvasWidth / 5 * 4;
        mh = canvasHeight / 4;

        options = new demo_opt();
        gui = new dat.GUI();

        stat.add("detector");

        //init interest points
        corners_img_u8 = new jsfeat.matrix_t(canvasWidth/2, canvasHeight, jsfeat.U8_t | jsfeat.C1_t);
        corners = [];
        var i = canvasWidth*canvasHeight;
        while(--i >= 0) {
            corners[i] = new jsfeat.keypoint_t(0,0,0,0);
        }
        var f2 = gui.addFolder('CORNERS');
        f2.add(options, 'threshold', 5, 100).step(1);
        f2.add(options, 'show_track_pts');
        f2.open();        

        //init LK tracker
        curr_img_pyr = new jsfeat.pyramid_t(3);
        prev_img_pyr = new jsfeat.pyramid_t(3);
        curr_img_pyr.allocate(canvasWidth, canvasHeight, jsfeat.U8_t|jsfeat.C1_t);
        prev_img_pyr.allocate(canvasWidth, canvasHeight, jsfeat.U8_t|jsfeat.C1_t);

        point_count = 0;
        point_status = new Uint8Array(MAX_POINTS);
        point_attr = new Int32Array(MAX_POINTS);
        prev_xy = new Float32Array(MAX_POINTS*2);
        curr_xy = new Float32Array(MAX_POINTS*2);

        var f3 = gui.addFolder('LK tracker');
        f3.add(options, 'win_size', 7, 30).step(1);
        f3.add(options, 'max_iterations', 3, 30).step(1);
        f3.add(options, 'epsilon', 0.001, 0.1).step(0.0025);
        f3.add(options, 'min_eigen', 0.001, 0.01).step(0.0025);
        f3.open();

        stat.add("optical flow");

        gui.close();
    }

    function detect_one() {
      //find corners
      //assume clean frame was drawn to canvas
      var imageData = ctx.getImageData(0, 0, canvasWidth/2, canvasHeight);
      jsfeat.imgproc.grayscale(imageData.data, canvasWidth/2, canvasHeight, corners_img_u8);
      jsfeat.fast_corners.set_threshold(options.threshold);
      var cornersCount = jsfeat.fast_corners.detect(corners_img_u8, corners, 5);

      if (cornersCount == 0) {
        return [];
      }

      var dim = 150;
      var tries = Math.min(5, cornersCount);
      var x, y;

      var good = false;

      //don't overlay existing objects
      while (tries > 0 && !good) {
        var idx = ~~(Math.random() * cornersCount);
        x = corners[idx].x;
        y = corners[idx].y;
        good = true;

        for (var key in faces) {
          var coords = faces[key].coords;
          if (
            x > coords.x - dim/2 && 
            x < coords.x + coords.w + dim/2 && 
            y > coords.y - dim/2 && 
            y < coords.y + coords.h + dim/2){
            good = false;
            break;
          }
        };
        tries--;
      }

      if (good) {
        return [{
          x: x,
          y: y,
          width: dim,
          height: dim,
          confidence: 200,
          neighbors: 0
        }];
      }
      return [];
    }

    function addTrackingPoints(face, faceIdx) {
        var pointsPerFace = 30;

        //push the center point
        curr_xy[point_count<<1] = ~~(face.coords.cx);
        curr_xy[(point_count<<1)+1] = ~~(face.coords.cy);
        point_attr[point_count] = faceIdx;
        face.points.push(point_count);
        point_count++;

        for (var j = 1; j < pointsPerFace; j++) {
            var r = Math.random() * Math.min(face.coords.w, face.coords.h) * 0.3;
            var th = Math.random()*2*Math.PI;

            curr_xy[point_count<<1] = ~~(r * Math.cos(th) + face.coords.cx);
            curr_xy[(point_count<<1)+1] = ~~(r * Math.sin(th) + face.coords.cy);
            point_attr[point_count] = faceIdx;
            face.points.push(point_count);
            point_count++;
        };
    }

    function track() {
        var imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);

        // swap flow data
        var _pt_xy = prev_xy;
        prev_xy = curr_xy;
        curr_xy = _pt_xy;
        var _pyr = prev_img_pyr;
        prev_img_pyr = curr_img_pyr;
        curr_img_pyr = _pyr;

        jsfeat.imgproc.grayscale(imageData.data, canvasWidth, canvasHeight, curr_img_pyr.data[0]);

        curr_img_pyr.build(curr_img_pyr.data[0], true);

        jsfeat.optical_flow_lk.track(prev_img_pyr, curr_img_pyr, prev_xy, curr_xy, point_count, options.win_size|0, options.max_iterations|0, point_status, options.epsilon, options.min_eigen);
        
        //prune_oflow_points(ctx);
        var n = point_count;
        var i=0, j=0;

        for(; i < n; ++i) {
            if(point_status[i] == 1) {
                if(j < i) {
                    curr_xy[j<<1] = curr_xy[i<<1];
                    curr_xy[(j<<1)+1] = curr_xy[(i<<1)+1];
                    point_attr[j] = point_attr[i];
                }
                ++j;
            }
        }
        point_count = j;

        //filter points in faces
        // for (var i = 0; i < faces.length; i++) {
        //     faces[i].points = [];
        // }
        var key;
        for (key in faces) {
            faces[key].points = [];
        }
        for (i=0; i < point_count; i++) {
            faces[point_attr[i]].points.push(i);
        };
        // faces = faces.filter(function(face){
        //   return face.points.length > 0;
        // });

        //remove faces with no points
        for (key in faces) {
            if (faces.hasOwnProperty(key) && faces[key].points.length == 0) {
                delete faces[key];
            }
            if (faces.hasOwnProperty(key) && !faces[key].is_live && faces[key].age > 20) {
              //dead for too long - remove
              //TODO: remove points
              // delete faces[key];
            }
        }
    }

    function track_fake2() {
      var key;
      for (key in faces) {
        var face = faces[key];
        var dx = face.coords.cx - face.old_coords.cx;
        var dy = face.coords.cy - face.old_coords.cy;
        dx /= 4;
        dy /= 4;
        for (var i = 0; i < face.points.length; i++) {
          var j = face.points[i];
          curr_xy[j<<1] += dx;
          curr_xy[(j<<1)+1] += dy;
        };
        setFaceCoords(face, face.coords.x + dx, face.coords.y + dy);
      }
    }

    function track_fake() {
      for (var key in faces) {
        var face = faces[key];
        var dx = face.coords.x - face.old_coords.x;
        var dy = face.coords.y - face.old_coords.y;
        dx /= 3;
        dy /= 3;
        setFaceCoords(face, face.coords.x + dx, face.coords.y + dy);
      }
    }

    function updateFaces() {
        for (var i in faces) {
            var face = faces[i];
            face.age++;

            var dx = 0;
            var dy = 0;
            
            if (face.points.length == 0) continue;

            var old_cx = 0,
                old_cy = 0,
                new_cx = 0,
                new_cy = 0;

            for (var j = 0; j < face.points.length; j++) {
                var idx = face.points[j];
                new_cx += curr_xy[idx<<1];
                new_cy += curr_xy[(idx<<1)+1];
            };

            new_cx /= face.points.length;
            new_cy /= face.points.length;

            setFaceCoords(face, new_cx - face.coords.w/2, new_cy - face.coords.h/2);
        };
    }

    /** 
    * Check if some of the faces were "eaten"
    * Update score and face state accordingly
    */
    function updateScore(mx, mh) {
        var mTop = (canvasHeight - mh)/2;
        var mBottom = canvasHeight - (canvasHeight - mh)/2
        for (var i in faces) {
          var face = faces[i];
          if (
            face.is_live &&
            face.old_coords.cx < mx && 
            face.coords.cx > mx &&
            face.old_coords.cy > mTop &&
            face.coords.cy > mTop &&
            face.old_coords.cy < mBottom &&
            face.coords.cy < mBottom) {
            score += face.score;
            face.is_live = false;
            face.age = 0;
          }
        };
    }

    var rects = [];

    function tick() {
        compatibility.requestAnimationFrame(tick);
        stat.new_frame();
        if (video.readyState === video.HAVE_ENOUGH_DATA) {

            // ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
            ctx.drawImage(video, 0, 0);
            var scale = 1;

            //track faces
            if (false && frames % 2 == 0){
              track_fake();
              // updateFaces();
            } else {
              // do track
              stat.start("optical flow");
              track();
              stat.stop("optical flow");
              
              updateFaces();
            }

            //add faces if needed
            if (Object.keys(faces).length < MIN_FACES){
              stat.start("detector");
              rects = detect_one();
              stat.stop("detector");
              lastDetectTime = Date.now();

              //update faces
              for (var i = 0; i < rects.length; i++) {
                  var type = Math.random() < 0.66 ? TYPE_FLOWER : TYPE_CHERRY;
                  var newFace = new Face(rects[i].x, rects[i].y, rects[i].width, rects[i].height, type, type+1);
                  addTrackingPoints(newFace, newFace.id);
                  faces[newFace.id] = newFace;
              };
            }

            updateScore(mx, mh);

            // draw_rects(ctx, rects, scale, 5);
            draw_faces(ctx, faces, scale, 5);
            if (options.show_track_pts) {
              draw_points(ctx, curr_xy, point_count);
            }
            draw_monster_diamond(ctx, mx, mh, scale);
            draw_score(ctx, score);

            frames++;
        }
        $('#log').html(stat.log());
    }

    function draw_points(ctx, points, count) {
        ctx.fillStyle = "rgb(0, 255, 0)";
        for (var i = 0; i < count; i++) {
            var x = points[i<<1];
            var y = points[(i<<1)+1];
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI*2, true);
            ctx.closePath();
            ctx.fill();
        };
    }

    function draw_rects(ctx, rects, sc, max) {
        ctx.strokeStyle = "rgb(0, 255, 0)";
        var on = rects.length;
        // if(on && max) {
        //     jsfeat.math.qsort(rects, 0, on-1, function(a,b){return (b.confidence<a.confidence);})
        // }
        var n = max || on;
        n = Math.min(n, on);
        var r;
        for(var i = 0; i < n; ++i) {
            r = rects[i];
            var size = 6;
            var x = (r.x*sc + r.width*sc*0.5 - size * 0.5)|0;
            var y = (r.y*sc + r.height*sc*0.5 - size * 0.5)|0;
            ctx.fillRect(x, y, size, size);
            ctx.strokeRect((r.x*sc)|0,(r.y*sc)|0,(r.width*sc)|0,(r.height*sc)|0);
        }
    }

    /**
    * Draw the monster
    */
    function draw_monster_round(ctx, mx, mh, sc) {
        var rad = mh * sc * 0.5;

        var openPct = (Date.now() - startTime) % 1000 * (360 / 1000 *2);
        openPct = (Math.sin(openPct * Math.PI / 180) + 1) / 2;
        ctx.fillStyle = "rgb(255, 255, 0)";
        //top half
        ctx.beginPath();
        ctx.arc(mx * sc, canvasHeight /2, rad, (0.25 - 0.25 * openPct) * Math.PI, (1.25 - 0.25 * openPct)* Math.PI, true);
        ctx.fill();
        //bottom half
        ctx.beginPath();
        ctx.arc(mx * sc, canvasHeight /2, rad, (0.75 + 0.25 * openPct)* Math.PI, (1.75 + 0.25 * openPct) * Math.PI, true);
        ctx.fill();

        //eye
        ctx.beginPath();
        ctx.arc(mx * sc, canvasHeight /2 - rad/2, 0.2*rad, 0, 2 * Math.PI, false);
        ctx.fillStyle = "rgb(0, 0, 0)";
        ctx.fill();


        // ctx.lineWidth="1";
        // ctx.strokeStyle = "rgb(0,0,255)";
        // ctx.strokeRect(mx * sc, (canvasHeight - mh * sc)/2, 6, mh * sc);
    }

    function draw_monster_diamond(ctx, mx, mh, sc) {
        var rad = mh * sc * 0.5;

        var openPct = (Date.now() - startTime) % 750 / 750;
        var ang = 40 * (Math.sin(openPct * 360 * Math.PI / 180) + 1) / 2;
        ctx.fillStyle = "rgb(255, 255, 0)";
        //TODO: translate + rotate canvas
        //top half
        var x0 = mx * sc + 2*rad, y0 = canvasHeight /2;
        ctx.lineWidth = 3;
        ctx.save();
        ctx.translate(x0, y0);
        ctx.rotate(0.5*ang * Math.PI / 180);
        //ctx.fillRect(0, 0, 10, 10);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-2*rad, 0);
        ctx.lineTo(-rad, - 0.866*rad);
        ctx.lineTo(0, 0);
        ctx.fill();

        //eye
        ctx.beginPath();
        ctx.arc(-1.1*rad, -0.5*rad, 0.2*rad, 0, 2 * Math.PI, false);
        ctx.fillStyle = "rgb(255,255,255)";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(-rad*1.2, -0.5*rad, 0.1*rad, 0, 2 * Math.PI, false);
        ctx.fillStyle = "rgb(0, 0, 0)";
        ctx.fill();

        //eye brow
        ctx.lineWidth = 5;
        ctx.strokeStyle = "rgb(0,0,0)";
        ctx.beginPath();
        ctx.moveTo(-1.4*rad, -0.5*rad);
        ctx.lineTo(-1.0*rad, -0.7*rad);
        ctx.stroke();

        
        //bottom half
        ctx.rotate(-1.5*ang * Math.PI / 180)
        ctx.fillStyle = "rgb(255, 255, 0)";
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-rad, 0.866*rad);
        ctx.lineTo(-2*rad, 0);
        ctx.lineTo(0, 0);
        ctx.fill();

        //tooth
        ctx.fillStyle = "rgb(255, 255, 255)";
        ctx.beginPath();
        ctx.moveTo(-1.5*rad, 0);
        ctx.lineTo(-1.6*rad, -0.433*rad);
        ctx.lineTo(-1.7*rad, 0);
        ctx.lineTo(0, 0);
        ctx.fill();


        ctx.restore();
    }

    /**
    * Draw the score
    */
    function draw_score(ctx, score) {
        ctx.font="30px Verdana";
        ctx.fillStyle = "rgb(255,255,255)";
        ctx.fillText("Score: " + score, 10, 40);
    }

    /**
    *
    * @param ctx canvas context to draw on
    * @param faces array of Face objects to draw
    * @param sc scale factor from working canvas to output canvas
    * @max   max number of rectangles to draw
    */
    function draw_faces(ctx, faces, sc) {
        ctx.fillStyle = "rgb(0,255,0)";

        for (var i in faces) {
          var face;
          face = faces[i];

          // Rescale coordinates from detector to video coordinate space:
          // var x = face.cx * video.videoWidth / canvasWidth;
          // var y = face.cy * video.videoHeight / canvasHeight;

          var size = 30;

          var cx = face.coords.cx * sc;
          var cy = face.coords.cy * sc;

          if (face.is_live) {
            if (face.type == TYPE_FLOWER) {
              draw_flower(ctx, face);
            } else {
              draw_cherry(ctx, face);
            }

          } else {
            ctx.lineWidth="5";
            ctx.strokeStyle = "rgb(255,0,0)";
            ctx.fillStyle = "rgb(255,0,0)";
            // ctx.beginPath();
            // ctx.moveTo(cx-size/2, cy-size/2);
            // ctx.lineTo(cx+size/2, cy+size/2);
            // ctx.moveTo(cx+size/2, cy-size/2);
            // ctx.lineTo(cx-size/2, cy+size/2);
            // ctx.stroke();

            ctx.fillText('+'+face.score, cx, cy - face.age*face.age);
          }
        }
    }

    function draw_flower(ctx, face) {
      ctx.lineWidth="1";
      // ctx.strokeStyle = "rgb(0,255,0)";
      // ctx.strokeRect(cx, cy, size, size);

      var cx = face.coords.cx;
      var cy = face.coords.cy;
      var size = 30;

      var rad = Math.min(face.age, 10);
      ctx.fillStyle = "rgb(255,100,255)";
      ctx.beginPath();
      ctx.arc(cx-1.5*rad, cy, rad, 0, Math.PI*2, true);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx+1.5*rad, cy, rad, 0, Math.PI*2, true);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx, cy-1.5*rad, rad, 0, Math.PI*2, true);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx, cy+1.5*rad, rad, 0, Math.PI*2, true);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = "rgb(0,0,0)";
      ctx.beginPath();
      ctx.moveTo(cx, cy-size/2);
      ctx.lineTo(cx, cy-5);
      ctx.moveTo(cx, cy+5);
      ctx.lineTo(cx, cy+size/2);

      ctx.moveTo(cx-size/2, cy);
      ctx.lineTo(cx-5, cy);
      ctx.moveTo(cx+5, cy);
      ctx.lineTo(cx+size/2, cy);
      ctx.stroke();
    }

    function draw_cherry(ctx, face) {
      var rad = Math.min(face.age, 10);

      var cx = face.coords.cx;
      var cy = face.coords.cy;

      ctx.strokeStyle = "rgb(192,128,0)";
      ctx.lineWidth="2";
      ctx.beginPath();
      ctx.moveTo(cx-1.5*rad, cy);
      ctx.lineTo(cx-1.5*rad, cy-rad*4);
      ctx.lineTo(cx+1.5*rad, cy);
      ctx.stroke();

      ctx.fillStyle = "rgb(255,0,0)";
      ctx.beginPath();
      ctx.arc(cx-1.5*rad, cy, rad, 0, Math.PI*2, true);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx+1.5*rad, cy, rad, 0, Math.PI*2, true);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = "rgb(255,255,255)";
      ctx.beginPath();
      ctx.arc(cx-1.8*rad, cy-0.5*rad, rad/4, 0, Math.PI*2, true);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx+1.2*rad, cy-0.5*rad, rad/4, 0, Math.PI*2, true);
      ctx.closePath();
      ctx.fill();
    }


    $(window).unload(function() {
        video.pause();
        video.src=null;
    });
});