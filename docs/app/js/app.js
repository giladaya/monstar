(function($, compatibility, profiler, jsfeat, dat) {
  "use strict";

  var MAX_POINTS = 200; //global max tracking points
  var MIN_TARGETS = 4; //min number of targets on screen (if lower -> get more)
  var TARGET_SIZE = 150; //width / height of a target in pixels
  var POINTS_PER_TARGET = 30; //max tracking points per target

  // target types
  var TYPE_FLOWER = 0;
  var TYPE_CHERRY = 1;

  var video = document.getElementById('webcam');
  var canvas = document.getElementById('canvas');

  var stat = new profiler();

  var gui, options, ctx, canvasWidth, canvasHeight;

  //interest point variables
  var corners_img_u8, corners;

  //tracking variables
  var curr_img_pyr, prev_img_pyr, point_count, point_status, prev_xy, curr_xy;

  var startTime = Date.now(); //game start time
  var frames = 0; //total frames drawn
  var score = 0; //current score
  var targets = {}; //target array
  var point_attr; //point to target lookup

  var mx; //moster x location
  var mh; //monster height


  //Try to lock screen orientation
  document.getElementById("btn_start").addEventListener("click", function() {
    if ('orientation' in screen && 'lock' in screen.orientation) {
      // document.documentElement.requestFullScreen();
      compatibility.requestFullScreen(document.documentElement);
      screen.orientation.lock("landscape-primary").then().catch(function(err) {
        notify('Failed to lock orientation');
        console.log(err);
      });
    }
    initVideo();
    document.getElementById('cover').className += ' hidden';
  }, false);

  
  /**
   * Initialize camera video stream
   */
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
          if (attempts < 10) {
            attempts++;
            setTimeout(findVideoSize, 200);
          } else {
            onDimensionsReady(640, 480);
          }
        }
      };
      var onDimensionsReady = function(width, height) {
        //start the app
        startApp(width, height);
        compatibility.requestAnimationFrame(tick);
      };

      video.addEventListener('loadeddata', readyListener);

      compatibility.getUserMedia({
          video: {
            facingMode: {
              exact: 'environment'
            }
          }
        },
        onGumSuccess,
        onGumError);
    } catch (error) {
      console.log(error);
      notify('Something went wrong...');
    }
  }

  /** 
   * getUserMedia callback
   */
  function onGumSuccess(stream) {
    try {
      video.src = compatibility.URL.createObjectURL(stream);
    } catch (error) {
      video.src = stream;
    }
    setTimeout(function() {
      video.play();
    }, 500);
  }

  /** 
   * getUserMedia callback
   */
  function onGumError(error) {
    notify('WebRTC not available.');
  }

  /** 
   * Notify of an error
   */
  function notify(msg) {
    $('#err').html(msg);
    $('#err').show();
  }
  

  /** 
   * Constructor for a Target object
   * @param cx target center x
   * @param cy target center y
   */
  function Target(x, y, w, h, type, score) {
    type = type || TYPE_FLOWER;
    score = score || 1;
    var coords = {
      'x': x,
      'y': y,
      'w': w,
      'h': h,
      'cx': x + w / 2,
      'cy': y + h / 2
    };
    return {
      'id': ~~(Math.random() * 10000000),
      'coords': coords,
      'old_coords': coords,
      'is_stale': false,
      'is_live': true,
      'points': [], //indexes of related tracking points
      'age': 0, //age since birth / since death
      'type': type, //object type
      'score': score //object score
    };
  }

  function setTargetCoords(target, x, y) {
    target.old_coords = target.coords;
    target.coords = Object.assign({}, target.coords, {
      x: x,
      y: y,
      cx: x + target.coords.w / 2,
      cy: y + target.coords.h / 2
    });
  }

  /**
   * runtime options
   */
  var Options = function() {
    //points
    this.threshold = 20;
    this.show_track_pts = false;

    //tracker
    this.win_size = 20;
    this.max_iterations = 30;
    this.epsilon = 0.01;
    this.min_eigen = 0.001;
  };

  /**
   * Initalize the app gui and data structures
   */
  function startApp(videoWidth, videoHeight) {
    canvasWidth = canvas.width;
    // canvasHeight = canvas.height;
    canvasHeight = ~~(canvas.width * window.innerHeight / window.innerWidth);
    canvas.height = canvasHeight;
    ctx = canvas.getContext('2d');

    //calculate monster location
    mx = canvasWidth / 5 * 4;
    mh = canvasHeight / 4;

    options = new Options();
    gui = new dat.GUI();


    //init interest points
    corners_img_u8 = new jsfeat.matrix_t(canvasWidth / 2, canvasHeight, jsfeat.U8_t | jsfeat.C1_t);
    corners = [];
    var i = canvasWidth * canvasHeight;
    while (--i >= 0) {
      corners[i] = new jsfeat.keypoint_t(0, 0, 0, 0);
    }
    var f2 = gui.addFolder('CORNERS');
    f2.add(options, 'threshold', 5, 100).step(1);
    f2.add(options, 'show_track_pts');
    f2.open();

    stat.add("detector");

    //init LK optical flow tracker
    curr_img_pyr = new jsfeat.pyramid_t(3);
    prev_img_pyr = new jsfeat.pyramid_t(3);
    curr_img_pyr.allocate(canvasWidth, canvasHeight, jsfeat.U8_t | jsfeat.C1_t);
    prev_img_pyr.allocate(canvasWidth, canvasHeight, jsfeat.U8_t | jsfeat.C1_t);

    point_count = 0;
    point_status = new Uint8Array(MAX_POINTS);
    point_attr = new Int32Array(MAX_POINTS);
    prev_xy = new Float32Array(MAX_POINTS * 2);
    curr_xy = new Float32Array(MAX_POINTS * 2);

    var f3 = gui.addFolder('LK tracker');
    f3.add(options, 'win_size', 7, 30).step(1);
    f3.add(options, 'max_iterations', 3, 30).step(1);
    f3.add(options, 'epsilon', 0.001, 0.1).step(0.0025);
    f3.add(options, 'min_eigen', 0.001, 0.01).step(0.0025);
    f3.open();

    stat.add("optical flow");

    gui.close();

    //register cleanup
    $(window).unload(function() {
      video.pause();
      video.src = null;
    });
  }

  /**
   * Find a new target:
   * 1. find interest points
   * 2. pick one in random
   * 3. if it does not ovelap any other target we're done, otherwise go to 2
   * 
   */
  function findNewTarget() {
    //find corners
    //assume that a clean frame was drawn to canvas
    var imageData = ctx.getImageData(0, 0, canvasWidth / 2, canvasHeight);
    jsfeat.imgproc.grayscale(imageData.data, canvasWidth / 2, canvasHeight, corners_img_u8);
    jsfeat.fast_corners.set_threshold(options.threshold);
    var cornersCount = jsfeat.fast_corners.detect(corners_img_u8, corners, 5);

    if (cornersCount === 0) {
      return [];
    }

    
    var tries = Math.min(5, cornersCount);
    var x, y;

    var good = false;

    //don't overlay existing targets
    while (tries > 0 && !good) {
      var idx = ~~(Math.random() * cornersCount);
      x = corners[idx].x;
      y = corners[idx].y;
      good = true;

      for (var key in targets) {
        var coords = targets[key].coords;
        if (
          x > coords.x - TARGET_SIZE / 2 &&
          x < coords.x + coords.w + TARGET_SIZE / 2 &&
          y > coords.y - TARGET_SIZE / 2 &&
          y < coords.y + coords.h + TARGET_SIZE / 2) {
          good = false;
          break;
        }
      }
      tries--;
    }

    if (good) {
      return [{
        x: x,
        y: y,
        width: TARGET_SIZE,
        height: TARGET_SIZE,
        confidence: 200,
        neighbors: 0
      }];
    }
    return [];
  }

  /**
   * add random tracking points to the target
   */
  function addTrackingPoints(target, targetIdx) {
    //push the center point
    curr_xy[point_count << 1] = ~~(target.coords.cx);
    curr_xy[(point_count << 1) + 1] = ~~(target.coords.cy);
    point_attr[point_count] = targetIdx;
    target.points.push(point_count);
    point_count++;

    for (var j = 1; j < POINTS_PER_TARGET; j++) {
      //pick a random point around the center
      var r = Math.random() * Math.min(target.coords.w, target.coords.h) * 0.3;
      var th = Math.random() * 2 * Math.PI;

      curr_xy[point_count << 1] = ~~(r * Math.cos(th) + target.coords.cx);
      curr_xy[(point_count << 1) + 1] = ~~(r * Math.sin(th) + target.coords.cy);
      point_attr[point_count] = targetIdx;
      target.points.push(point_count);
      point_count++;
    }
  }

  /**
   * track all tracking points using LK optical flow algorithm
   */
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

    jsfeat.optical_flow_lk.track(prev_img_pyr, curr_img_pyr, prev_xy, curr_xy, point_count, options.win_size | 0, options.max_iterations | 0, point_status, options.epsilon, options.min_eigen);

    //prune overflow points
    var n = point_count;
    var i = 0,
      j = 0;

    for (; i < n; ++i) {
      if (point_status[i] == 1) {
        if (j < i) {
          curr_xy[j << 1] = curr_xy[i << 1];
          curr_xy[(j << 1) + 1] = curr_xy[(i << 1) + 1];
          point_attr[j] = point_attr[i];
        }
        ++j;
      }
    }
    point_count = j;

    //filter points in targets
    var key;
    for (key in targets) {
      targets[key].points = [];
    }
    for (i = 0; i < point_count; i++) {
      targets[point_attr[i]].points.push(i);
    }

    //remove targets with no points
    for (key in targets) {
      if (targets.hasOwnProperty(key) && targets[key].points.length === 0) {
        delete targets[key];
      }
      if (targets.hasOwnProperty(key) && !targets[key].is_live && targets[key].age > 20) {
        //dead for too long - remove
        //TODO: remove points
        // delete targets[key];
      }
    }
  }

  /**
   * Fake tracking function. 
   * Used once every other frame to lower the CPU load
   * Moves the target a little along the same vector as the last movement
   */
  function track_fake() {
    for (var key in targets) {
      var target = targets[key];
      var dx = target.coords.x - target.old_coords.x;
      var dy = target.coords.y - target.old_coords.y;
      dx /= 3;
      dy /= 3;
      setTargetCoords(target, target.coords.x + dx, target.coords.y + dy);
    }
  }

  /**
   * Move the targets to the new location according to the optical flow
   */
  function updateTargets() {
    for (var i in targets) {
      var target = targets[i];
      target.age++;

      if (target.points.length === 0) continue;

      var new_cx = 0,
          new_cy = 0;

      //calculate centroid
      for (var j = 0; j < target.points.length; j++) {
        var idx = target.points[j];
        new_cx += curr_xy[idx << 1];
        new_cy += curr_xy[(idx << 1) + 1];
      }

      new_cx /= target.points.length;
      new_cy /= target.points.length;

      setTargetCoords(target, new_cx - target.coords.w / 2, new_cy - target.coords.h / 2);
    }
  }

  /** 
   * Check if some of the targets were "eaten"
   * Update score and target state accordingly
   */
  function updateScore(mx, mh) {
    var mTop = (canvasHeight - mh) / 2;
    var mBottom = canvasHeight - (canvasHeight - mh) / 2;
    for (var i in targets) {
      var target = targets[i];
      if (
        target.is_live &&
        target.old_coords.cx < mx &&
        target.coords.cx > mx &&
        target.old_coords.cy > mTop &&
        target.coords.cy > mTop &&
        target.old_coords.cy < mBottom &&
        target.coords.cy < mBottom) {
        score += target.score;
        target.is_live = false;
        target.age = 0;
      }
    }
  }


  function tick() {
    compatibility.requestAnimationFrame(tick);
    stat.new_frame();
    if (video.readyState === video.HAVE_ENOUGH_DATA) {

      // ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
      ctx.drawImage(video, 0, 0);
      var scale = 1;

      //track targets
      if (false && frames % 2 === 0) {
        track_fake();
        // updateTargets();
      } else {
        // do track
        stat.start("optical flow");
        track();
        stat.stop("optical flow");

        updateTargets();
      }

      var rects = [];
      //add targets if needed
      if (Object.keys(targets).length < MIN_TARGETS) {
        stat.start("detector");
        rects = findNewTarget();
        stat.stop("detector");

        //update targets
        for (var i = 0; i < rects.length; i++) {
          var type = Math.random() < 0.66 ? TYPE_FLOWER : TYPE_CHERRY;
          var newTarget = new Target(rects[i].x, rects[i].y, rects[i].width, rects[i].height, type, type + 1);
          addTrackingPoints(newTarget, newTarget.id);
          targets[newTarget.id] = newTarget;
        }
      }

      updateScore(mx, mh);

      draw_targets(ctx, targets, scale, 5);
      if (options.show_track_pts) {
        draw_points(ctx, curr_xy, point_count);
      }
      draw_monster_diamond(ctx, mx, mh, scale);
      draw_score(ctx, score);

      frames++;
    }
    showStats();
  }

  function showStats() {
    $('#log').html(stat.log());
  }

  /**
   * Draw the tracking points
   */
  function draw_points(ctx, points, count) {
    ctx.fillStyle = "rgb(0, 255, 0)";
    for (var i = 0; i < count; i++) {
      var x = points[i << 1];
      var y = points[(i << 1) + 1];
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2, true);
      ctx.closePath();
      ctx.fill();
    }
  }

  /**
   * Draw the monster (round version)
   *
   * @param ctx canvas context 
   * @param mx monster x loacation 
   * @param mh monster height 
   * @param sc scale 
   */
  function draw_monster_round(ctx, mx, mh, sc) {
    sc = sc || 1.0;
    var rad = mh * sc * 0.5;

    var openPct = (Date.now() - startTime) % 1000 * (360 / 1000 * 2);
    openPct = (Math.sin(openPct * Math.PI / 180) + 1) / 2;
    ctx.fillStyle = "rgb(255, 255, 0)";
    //top half
    ctx.beginPath();
    ctx.arc(mx * sc, canvasHeight / 2, rad, (0.25 - 0.25 * openPct) * Math.PI, (1.25 - 0.25 * openPct) * Math.PI, true);
    ctx.fill();
    //bottom half
    ctx.beginPath();
    ctx.arc(mx * sc, canvasHeight / 2, rad, (0.75 + 0.25 * openPct) * Math.PI, (1.75 + 0.25 * openPct) * Math.PI, true);
    ctx.fill();

    //eye
    ctx.beginPath();
    ctx.arc(mx * sc, canvasHeight / 2 - rad / 2, 0.2 * rad, 0, 2 * Math.PI, false);
    ctx.fillStyle = "rgb(0, 0, 0)";
    ctx.fill();
  }

  /**
   * Draw the monster (diamond version)
   *
   * @param ctx canvas context 
   * @param mx monster x loacation 
   * @param mh monster height 
   * @param sc scale
   */
  function draw_monster_diamond(ctx, mx, mh, sc) {
    sc = sc || 1.0;
    var rad = mh * sc * 0.5;

    var openPct = (Date.now() - startTime) % 750 / 750;
    var ang = 40 * (Math.sin(openPct * 360 * Math.PI / 180) + 1) / 2;
    ctx.fillStyle = "rgb(255, 255, 0)";

    var x0 = mx * sc + 2 * rad,
      y0 = canvasHeight / 2;
    ctx.lineWidth = 3;
    ctx.save();
    ctx.translate(x0, y0);
    ctx.rotate(0.5 * ang * Math.PI / 180);

    //top half
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-2 * rad, 0);
    ctx.lineTo(-rad, -0.866 * rad);
    ctx.lineTo(0, 0);
    ctx.fill();

    //eye
    ctx.beginPath();
    ctx.arc(-1.1 * rad, -0.5 * rad, 0.2 * rad, 0, 2 * Math.PI, false);
    ctx.fillStyle = "rgb(255,255,255)";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(-rad * 1.2, -0.5 * rad, 0.1 * rad, 0, 2 * Math.PI, false);
    ctx.fillStyle = "rgb(0, 0, 0)";
    ctx.fill();

    //eye brow
    ctx.lineWidth = 5;
    ctx.strokeStyle = "rgb(0,0,0)";
    ctx.beginPath();
    ctx.moveTo(-1.4 * rad, -0.5 * rad);
    ctx.lineTo(-1.0 * rad, -0.7 * rad);
    ctx.stroke();


    //bottom half
    ctx.rotate(-1.5 * ang * Math.PI / 180);
    ctx.fillStyle = "rgb(255, 255, 0)";
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-rad, 0.866 * rad);
    ctx.lineTo(-2 * rad, 0);
    ctx.lineTo(0, 0);
    ctx.fill();

    //tooth
    ctx.fillStyle = "rgb(255, 255, 255)";
    ctx.beginPath();
    ctx.moveTo(-1.5 * rad, 0);
    ctx.lineTo(-1.6 * rad, -0.433 * rad);
    ctx.lineTo(-1.7 * rad, 0);
    ctx.lineTo(0, 0);
    ctx.fill();

    ctx.restore();
  }

  /**
   * Draw the score
   */
  function draw_score(ctx, score) {
    ctx.font = "30px Verdana";
    ctx.fillStyle = "rgb(255,255,255)";
    ctx.fillText("Score: " + score, 10, 40);
  }

  /**
   * Draw the targets 
   * 
   * @param ctx canvas context to draw on
   * @param targets array of Target objects to draw
   * @param sc scale factor
   */
  function draw_targets(ctx, targets, sc) {
    ctx.fillStyle = "rgb(0,255,0)";

    for (var i in targets) {
      var target;
      target = targets[i];

      if (target.is_live) {
        if (target.type == TYPE_FLOWER) {
          draw_flower(ctx, target);
        } else {
          draw_cherry(ctx, target);
        }
      } else {
        var cx = target.coords.cx * sc;
        var cy = target.coords.cy * sc;
        ctx.lineWidth = "5";
        ctx.strokeStyle = "rgb(255,255,255)";
        ctx.fillStyle = "rgb(255,0,0)";
        ctx.strokeText('+' + target.score, cx, cy - target.age * target.age);
        ctx.fillText('+' + target.score, cx, cy - target.age * target.age);
      }
    }
  }

  /**
   * Draw a flower
   */
  function draw_flower(ctx, target) {
    ctx.lineWidth = "1";

    var cx = target.coords.cx;
    var cy = target.coords.cy;
    var size = 30;

    var rad = Math.min(target.age, 10);
    ctx.fillStyle = "rgb(255,100,255)";
    ctx.beginPath();
    ctx.arc(cx - 1.5 * rad, cy, rad, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + 1.5 * rad, cy, rad, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy - 1.5 * rad, rad, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy + 1.5 * rad, rad, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "rgb(255, 255, 255)";
    ctx.beginPath();
    ctx.moveTo(cx, cy - size / 2);
    ctx.lineTo(cx, cy - 5);
    ctx.moveTo(cx, cy + 5);
    ctx.lineTo(cx, cy + size / 2);

    ctx.moveTo(cx - size / 2, cy);
    ctx.lineTo(cx - 5, cy);
    ctx.moveTo(cx + 5, cy);
    ctx.lineTo(cx + size / 2, cy);
    ctx.stroke();
  }

  /**
   * Draw a pair of cherries
   */
  function draw_cherry(ctx, target) {
    var rad = Math.min(target.age, 10);

    var cx = target.coords.cx;
    var cy = target.coords.cy;

    ctx.strokeStyle = "rgb(192,128,0)";
    ctx.lineWidth = "2";
    ctx.beginPath();
    ctx.moveTo(cx - 1.5 * rad, cy);
    ctx.lineTo(cx - 1.5 * rad, cy - rad * 4);
    ctx.lineTo(cx + 1.5 * rad, cy);
    ctx.stroke();

    ctx.fillStyle = "rgb(255,0,0)";
    ctx.beginPath();
    ctx.arc(cx - 1.5 * rad, cy, rad, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + 1.5 * rad, cy, rad, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = "rgb(255,255,255)";
    ctx.beginPath();
    ctx.arc(cx - 1.8 * rad, cy - 0.5 * rad, rad / 4, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + 1.2 * rad, cy - 0.5 * rad, rad / 4, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fill();
  }
  
})($, compatibility, profiler, jsfeat, dat);