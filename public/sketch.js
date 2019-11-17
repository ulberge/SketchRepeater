(function() {
  const canvasSizeX = 300;
  const canvasSizeY = 250;

  const selectionSizes = [15, 45, 81, 105, 129];
  // const selectionColors = ['#A74661', '#956C89', '#8490B0', '#71B6D7', '#67CBEF'];
  const selectionColors = ['#A74661', '#956C89', '#8490B0', '#71B6D7', '#67CBEF'];
  const sketches = {
    stored: null,
    temp: null,
    compMarks: null,
    humanMarks: null,
    allPurpose: null,
    selections: {
      p0: [],
      p1: []
    },
    ai: [],
    ai_overlay: [],
    ai_debug: []
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
  const lastMarkBounds = [[], [], [], [], []];

  // Given a container and some dataURLs, clear the container and add the dataURLs as images to that container
  function setContentsToDataURLs(container, dataUrls) {
    container.empty();
    dataUrls.forEach(dataURL => {
      const img = new Image();
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
    const stridesX = Math.max(0, Math.ceil((boundsWidth - fSize) / stride));
    const stridesY = Math.max(0, Math.ceil((boundsHeight - fSize) / stride));
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

  function getActions(befores, mark, afters, imgs, bounds, n=1) {
    console.log('fetching actions...');
    const data = { befores, mark, afters, imgs, bounds, n };

    $.get('/actions', data, function(result) {
      console.log(result);
      const layers = result;

      for (let i = 0; i < 5; i += 1) {
        const layer = layers[i];
        const debug = sketches.ai_debug[i];
        const overlay = sketches.ai_overlay[i];

        overlay.clear();
        debug.clear();

        const { locationImgs, locations, actions } = layer;
        const location = locations[0];
        const action = actions[0];

        // Draw rectangle highlighting area selected for change
        const { x, y } = location;
        const selectionBounds = getSelectionBoundsForLayer(bounds, i);
        const w = selectionBounds[2] - selectionBounds[0];
        const h = selectionBounds[3] - selectionBounds[1];
        debug.stroke(debug.color(selectionColors[i]));
        debug.strokeWeight(2);
        debug.noFill();
        debug.rect(x, y, w, h);

        // Trace out lines on overlay
        const numLines = 4;
        const speed = 10;
        // Set bounds as the location where this mark was made and as the same size as the previous mark
        const boundsSize = Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1]);
        lastMarkBounds[i] = [x, y, x + boundsSize, y + boundsSize];
        lineTracer.trace(overlay, location, action, numLines, speed);
      }
      console.log(JSON.stringify(lastMarkBounds));

      // const container = $('#mark_suggestions');
      // container.empty();
      // layers[0].marks.forEach(dataURL => {
      //   const img = new Image();
      //   img.src = 'data:image/png;base64,' + dataURL;
      //   container.append(img);
      // });
    });
  }

  function onChange(bounds) {
    console.log('Edit with bounds ' + bounds);
    // get previous state of change area
    const selections_p0 = getChangeSelections(sketches.stored, bounds);
    const selections_mark = getChangeSelections(sketches.temp, bounds);
    const mark = sketches.temp.get(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]);

    const markDataUrl = [sketches.allPurpose].map(p => {
      p.resizeCanvas(mark.width, mark.height);
      p.background(255);
      p.image(mark, 0, 0);
      const dataURL = p.canvas.toDataURL();
      return dataURL;
    })[0];

    // write temp to stored
    const temp = sketches.temp.get();
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

    // Copy to temp and trigger change
    sketches.temp.image(mark, 0, 0);
    onChange(lastMarkBounds[i]);
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

  // Add extra canvases for other uses
  sketches.compMarks = new p5(getSketch(), document.getElementById('sketch_comp_marks'));
  sketches.humanMarks = new p5(getSketch(), document.getElementById('sketch_human_marks'));
  sketches.allPurpose = new p5(getSketch(), document.getElementById('sketch_allPurpose'));

  function sketch_stored(p) {
    sketches.stored = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.background(255);
      p.noLoop();
    };

    p.draw = function draw() {
      p.line(0, p.height / 1.5, p.width, p.height / 1.5);
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
          const pad = 10;
          bounds = [Math.max(0, bounds[0] - pad), Math.max(0, bounds[1] - pad), Math.min(p.width - 1, bounds[2] + pad), Math.min(p.height - 1, bounds[3] + pad)];
          onChange(bounds);
        }
        bounds = null;
      }
    };
  }
  new p5(sketch_temp, document.getElementById('sketch_temp'));
}());
