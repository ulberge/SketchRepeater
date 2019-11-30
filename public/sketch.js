(function() {
  const canvasSizeX = 300;
  const canvasSizeY = 250;

  const selectionSizes = [15, 45, 81, 105, 129];
  // const selectionColors = ['#A74661', '#956C89', '#8490B0', '#71B6D7', '#67CBEF'];
  // const selectionColors = ['#A74661', '#956C89', '#8490B0', '#71B6D7', '#67CBEF'];
  const selectionColors = ['#67CBEF', '#67CBEF', '#67CBEF', '#67CBEF', '#67CBEF'];
  const sketches = {
    stored: null,
    temp: null,
    compMarks: null,
    humanMarks: null,
    allPurpose: null,
    marksRecord: null,
    selections: {
      p0: [],
      p1: []
    },
    ai: [],
    ai_overlay: [],
    ai_debug: [],
  };
  let drawTimer = null;

  const layers_meta = [
    [ 'conv1', 3, 15, 0 ], // layer_name, stride, f_size, padding
    [ 'conv2', 6, 45, 0 ],
    [ 'conv3', 12, 81, 12 ],
    [ 'conv4', 12, 105, 24 ],
    [ 'conv5', 12, 129, 36 ]
  ];

  // Store the bounds of the most recent mark for each AI
  const lastBounds = [[], [], [], [], []];
  const lastMarkBounds = [[], [], [], [], []];

  // Given a container and some dataURLs, clear the container and add the dataURLs as images to that container
  function setContentsToDataURLs(container, dataUrls) {
    container.empty();
    dataUrls.forEach(dataURL => {
      const img = new Image();
      if (!dataURL.includes('data:image/png;base64,')) {
        dataURL = 'data:image/png;base64,' + dataURL;
      }
      img.src = dataURL;
      container.append(img);
    });
  }

  // Given bounds, find an appropriate size that will work nicely for this layer
  function getSelectionBoundsForLayer(bounds, i) {
    const [ bStartX, bStartY, bEndX, bEndY ] = bounds.map(bound => Math.floor(bound));
    const boundsWidth = bEndX - bStartX;
    const boundsHeight = bEndY - bStartY;

    const selectionSize = selectionSizes[i];
    const [ layerName, stride, fSize, padding ] = layers_meta[i];
    // How many full strides at this layer do you need to add to the smallest filter to get a section of the image that
    // will fit the change bounds
    const stridesX = Math.max(0, Math.floor((boundsWidth - fSize) / stride)) + Math.floor(i / 2);
    const stridesY = Math.max(0, Math.floor((boundsHeight - fSize) / stride)) + Math.floor(i / 2);
    // Use the max, get a square
    const sizeToSelect = Math.max(fSize + (stridesX * stride), fSize + (stridesY * stride));
    const horPad = (sizeToSelect - boundsWidth) / 2;
    const vertPad =(sizeToSelect - boundsHeight) / 2;

    const selectionBounds = [bStartX - horPad, bStartY - vertPad, bEndX + horPad, bEndY + vertPad];
    return selectionBounds;
  }

  function getChangeSelections(p, bounds) {
    // get smallest bounds of activation selection area
    const selectionBoundsBySize = selectionSizes.map((selectionSize, i) => {
      return getSelectionBoundsForLayer(bounds, i);
    });

    const selections = selectionBoundsBySize.map(selectionBounds => {
      const [ bStartX, bStartY, bEndX, bEndY ] = selectionBounds;
      return p.get(bStartX, bStartY, bEndX - bStartX, bEndY - bStartY);
    });
    return selections;
  }

  let toDraw = 0;
  async function draw(p, position, dataURL, numLines=3, speed=10, padding=0) {
    toDraw += 1;
    await lineTracer.trace(p, position, dataURL, numLines, speed, padding);
    toDraw -= 1;

    // if (toDraw === 0) {
    //   $('.change_imgs').removeClass('hidden');
    //   $('.debug_mark').removeClass('hidden');
    //   $('#marks_record').removeClass('hidden');
    //   html2canvas(document.querySelector('#view')).then(canvas => {
    //     $('#screenshot').append(canvas);
    //   });
    //   $('.change_imgs').addClass('hidden');
    //   $('.debug_mark').addClass('hidden');
    //   $('#marks_record').addClass('hidden');
    // }
  }

  function getActions(befores, mark, afters, imgs, bounds, n=1) {
    console.log('fetching actions...');
    const data = { befores, mark, afters, imgs, bounds, n };
    const befores_prev = befores;

    $.get('/actions', data, function(result) {
      const layers = result;

      setContentsToDataURLs($('#mark_matches'), layers[0].marks);

      for (let i = 0; i < 4; i += 1) {
        const layer = layers[i];
        const debug = sketches.ai_debug[i];
        const overlay = sketches.ai_overlay[i];

        overlay.clear();
        debug.clear();

        const { location, before, mark, after, befores } = layer;

        // Draw rectangle highlighting area selected for change
        const { x, y } = location;
        const selectionBounds = getSelectionBoundsForLayer(bounds, i);
        const w = selectionBounds[2] - selectionBounds[0];
        const h = selectionBounds[3] - selectionBounds[1];
        // debug.stroke(debug.color(selectionColors[i]));
        // debug.strokeWeight(2);
        // debug.noFill();
        // debug.rect(x, y, w, h);

        // Trace out lines on overlay
        const numLines = 4;
        const speed = 0.00001;
        draw(overlay, location, mark, numLines, speed, 12);

        // Draw debug stuff
        setContentsToDataURLs($('#ai' + i + '_debug_mark'), [before, mark, after]);
        // setContentsToDataURLs($('#ai' + i + '_debug_before'), befores);
        // setContentsToDataURLs($('#ai' + i + '_debug_before2'), [befores_prev[i]]);
        lastBounds[i] = [x, y, x + w, y + h];
        lastMarkBounds[i] = [x, y, x + w, y + h];
      }
      // const container = $('#mark_suggestions');
      // container.empty();
      // layers[0].marks.forEach(dataURL => {
      //   const img = new Image();
      //   img.src = 'data:image/png;base64,' + dataURL;
      //   container.append(img);
      // });
    });
  }

  function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
  }

  function onChange(bounds, isAI=false) {
    console.log('Edit with bounds ' + bounds);
    // get previous state of change area
    const selections_p0 = getChangeSelections(sketches.stored, bounds);
    const mark = sketches.temp.get(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]);

    const markDataUrl = [sketches.allPurpose].map(p => {
      const markSize = Math.max(mark.width, mark.height);
      p.resizeCanvas(markSize, markSize);
      p.background(255);
      p.image(mark, (markSize - mark.width) / 2, (markSize - mark.height) / 2);
      const dataURL = p.canvas.toDataURL();
      return dataURL;
    })[0];

    const temp = sketches.temp.get();

    // Create "human copy"
    sketches.ai[4].image(temp, 0, 0);

    const prev_store = sketches.s1_canv.get();
    sketches.s0_canv.image(prev_store, 0, 0);

    const stored = sketches.stored.get();
    sketches.s1_canv.image(stored, 0, 0);

    // rearrange
    const shuff = shuffleArray($('.ai_sketches > div'));
    $('.ai_sketches').append(shuff);
    $('.main_canvas').hide();
    html2canvas(document.querySelector('#view')).then(canvas => {
      $('#screenshot').append(canvas);
    });
    $('.main_canvas').show();

    // write temp to record
    // tint first to indicate source
    sketches.allPurpose.clear();
    sketches.allPurpose.resizeCanvas(sketches.temp.width, sketches.temp.height);
    sketches.allPurpose.image(temp, 0, 0);
    sketches.allPurpose.loadPixels();
    if (isAI) {
      sketches.allPurpose.pixels.forEach((v, i) => {
        if ((i % 4) === 1 || (i % 4) === 2) {
          sketches.allPurpose.pixels[i] = 120;
        }
      });
    } else {
      sketches.allPurpose.pixels.forEach((v, i) => {
        if ((i % 4) === 0) {
          sketches.allPurpose.pixels[i] = 165;
        }
        if ((i % 4) === 1) {
          sketches.allPurpose.pixels[i] = 80;
        }
      });
    }
    sketches.allPurpose.updatePixels();
    const tinted = sketches.allPurpose.get();
    sketches.marksRecord.image(tinted, 0, 0);

    // write temp to stored
    sketches.stored.image(temp, 0, 0);
    sketches.ai.forEach(p => {
      p.image(temp, 0, 0)
    });
    sketches.temp.clear();

    // get new state of change area
    const selections_p1 = getChangeSelections(sketches.stored, bounds);

    // render to displays and retrieve dataURLs (format for API)
    const befores = selections_p0.map((s, i) => {
      const p = sketches.selections.p0[i];
      p.resizeCanvas(s.width, s.height);
      p.background(255);
      p.image(s, 0, 0);
      const dataURL = p.canvas.toDataURL();
      return dataURL;
    });
    const afters = selections_p1.map((s, i) => {
      const p = sketches.selections.p1[i];
      p.resizeCanvas(s.width, s.height);
      p.background(255);
      p.image(s, 0, 0);
      const dataURL = p.canvas.toDataURL();
      return dataURL;
    });

    // draw to display imgs
    setContentsToDataURLs($('.change_imgs .p0'), [befores[4]]);
    setContentsToDataURLs($('.change_imgs .marks'), [markDataUrl]);
    setContentsToDataURLs($('.change_imgs .p1'), [afters[4]]);

    // for each ai canvas, get suggestions at the provided layers
    const imgs = sketches.ai.map(p => p.canvas.toDataURL());

    getActions(befores, markDataUrl, afters, imgs, bounds);
  }

  function copyStoredToAI() {
    const img = sketches.stored.get();
    sketches.ai.forEach(p => p.image(img, 0, 0));
  }

  function selectAI(i) {
    // Get the mark(s) for this AI
    const mark = sketches.ai_overlay[i].get();

    // Make reddish mark black
    sketches.allPurpose.clear();
    sketches.allPurpose.resizeCanvas(sketches.temp.width, sketches.temp.height);
    sketches.allPurpose.image(mark, 0, 0);
    sketches.allPurpose.loadPixels();
    sketches.allPurpose.pixels.forEach((v, i) => {
      if ((i % 4) === 0 && sketches.allPurpose.pixels[i] !== 0) {
        sketches.allPurpose.pixels[i] = 0;
        sketches.allPurpose.pixels[i + 1] = 0;
        sketches.allPurpose.pixels[i + 2] = 0;
      }
    });
    sketches.allPurpose.updatePixels();
    const markBlack = sketches.allPurpose.get();

    // Copy to temp and trigger change
    sketches.temp.image(markBlack, 0, 0);

    const w0 = lastBounds[0][2] - lastBounds[0][0];
    const h0 = lastBounds[0][3] - lastBounds[0][1];
    const wi = lastBounds[i][2] - lastBounds[i][0];
    const hi = lastBounds[i][3] - lastBounds[i][1];
    const diffW = wi - w0;
    const diffH = hi - h0;

    const startX = lastBounds[i][0] + Math.floor(diffW / 2);
    const startY = lastBounds[i][1] + Math.floor(diffH / 2);
    const newBounds = [startX, startY, startX + w0, startY + h0];

    onChange(newBounds, true);
  }

  function getSketch() {
    return (p) => {
      p.setup = function setup() {
        p.pixelDensity(1);
        p.createCanvas(canvasSizeX, canvasSizeY);
        p.noLoop();
      };
    };
  }

  function getSelectionSketch(layer_i, p_i) {
    return (p) => {
      sketches.selections['p' + p_i][layer_i - 1] = p;
      p.setup = function setup() {
        const sketchSize = selectionSizes[layer_i - 1];
        p.pixelDensity(1);
        p.createCanvas(sketchSize, sketchSize);
        p.background(255);
        p.noLoop();
      };

      p.draw = function draw() {
      };
    };
  }

  // For each layer
  for (let i = 0; i < 5; i += 1) {
    const containerId = '#L' + (i + 1) + '_matches';
    new p5(getSelectionSketch((i + 1), 0), $(containerId + ' .selection.p0')[0]);
    new p5(getSelectionSketch((i + 1), 1), $(containerId + ' .selection.p1')[0]);

    sketches.ai.push(new p5(getSketch(), document.getElementById('sketch_ai' + i)));
    sketches.ai_debug.push(new p5(getSketch(), document.getElementById('sketch_ai' + i + '_debug')));
    sketches.ai_overlay.push(new p5(getSketch(), document.getElementById('sketch_ai' + i + '_overlay')));
    $('#sketch_ai' + i + '_debug').click(() => selectAI(i));
  }
  // sketches.ai.push(new p5(getSketch(), document.getElementById('sketch_ai4')));

  sketches.s0_canv = new p5(getSketch(), document.getElementById('s0_canv'));
  sketches.s1_canv = new p5(getSketch(), document.getElementById('s1_canv'));

  // Add extra canvases for other uses
  sketches.compMarks = new p5(getSketch(), document.getElementById('sketch_comp_marks'));
  sketches.humanMarks = new p5(getSketch(), document.getElementById('sketch_human_marks'));
  sketches.allPurpose = new p5(getSketch(), document.getElementById('sketch_allPurpose'));

  sketches.marksRecord = new p5(getSketch(), document.getElementById('marks_record'));

  function sketch_stored(p) {
    sketches.stored = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.background(255);
      p.noLoop();
    };

    p.draw = function draw() {
      // p.line(0, p.height / 1.5, p.width, p.height / 1.5);
      copyStoredToAI();
    };
  }
  new p5(sketch_stored, document.getElementById('sketch_stored'));

  function sketch_temp(p) {
    sketches.temp = p;
    let bounds = null;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
    };

    p.draw = function draw() {
      p.fill(0);
      p.stroke(0);
      p.strokeWeight(1.3);

      // Record mouse pressed events within canvas
      const px = p.pmouseX;
      const py = p.pmouseY;
      const x = p.mouseX;
      const y = p.mouseY;
      if (!(x < 0 || y < 0 || px < 0 || py < 0 || x >= p.width || px >= p.width || y >= p.height || py >= p.height)) {
        if (p.mouseIsPressed) {
          // draw line
          p.line(px, py, x, y);

          // update bounds of this edit
          if (!bounds) {
            let minX = Math.min(px, x);
            let minY = Math.min(py, y);
            let maxX = Math.max(px, x);
            let maxY = Math.max(py, y);
            bounds = [minX, minY, maxX, maxY];
          } else {
            bounds = [Math.min(bounds[0], x), Math.min(bounds[1], y), Math.max(bounds[2], x), Math.max(bounds[3], y)];
          }
        }
      }
      // If mouse is not pressed, and it was being pressed at the last draw, trigger on change and clear
      if (!p.mouseIsPressed) {
        if (bounds) {
          const markPad = 10;
          const markPadAdjust = Math.min(bounds[0], bounds[1], p.width - bounds[2], p.height - bounds[3], markPad);
          const markBounds = [bounds[0] - markPadAdjust, bounds[1] - markPadAdjust, bounds[2] + markPadAdjust, bounds[3] + markPadAdjust];
          // grow the pads up to 45, if not near border
          // const pad = 45;
          // const padAdjust = Math.min(bounds[0], bounds[1], p.width - bounds[2], p.height - bounds[3], pad);
          // const beforeBounds = [bounds[0] - padAdjust, bounds[1] - padAdjust, bounds[2] + padAdjust, bounds[3] + padAdjust];

          onChange(markBounds);
        }
        bounds = null;
      }
    };
  }
  new p5(sketch_temp, document.getElementById('sketch_temp'));

  async function pause(t) {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve();
      }, t);
    });
  }

  async function downloadCanvases() {
    const date = Date.now();
    const canvases = $('#screenshot canvas');

    for (let i = 0; i < canvases.length; i += 1) {
      console.log(i);
      let downloadLink = document.createElement('a');
      downloadLink.setAttribute('download', 'survey_image' + date + '_' + i + '.png');
      let dataURL = canvases[i].toDataURL('image/png');
      let url = dataURL.replace(/^data:image\/png/,'data:application/octet-stream');
      downloadLink.setAttribute('href', url);
      downloadLink.click();
      await pause(100);
    }
  }

  document.addEventListener('keydown', e => {
    // if (e.key === 'a') {
    //   if ($('#marks_record').hasClass('hidden')) {
    //     $('#marks_record').removeClass('hidden');
    //   } else {
    //     $('#marks_record').addClass('hidden');
    //   }
    // }

    if (e.key === 'q') {
      if ($('.change_imgs').hasClass('hidden')) {
        $('.change_imgs').removeClass('hidden');
        $('.debug_mark').removeClass('hidden');
        $('#marks_record').removeClass('hidden');
      } else {
        $('.change_imgs').addClass('hidden');
        $('.debug_mark').addClass('hidden');
        $('#marks_record').addClass('hidden');
      }
    }

    if (e.key === 's') {
      // save record
      downloadCanvases();
    }
  });
  $('.change_imgs').addClass('hidden');
  $('.debug_mark').addClass('hidden');
  $('#marks_record').addClass('hidden');
}());
