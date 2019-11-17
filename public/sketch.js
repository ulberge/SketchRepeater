(function() {
  const canvasSizeX = 300;
  const canvasSizeY = 250;

  const selectionSizes = [15, 45, 81, 105, 129];
  // const selectionColors = ['#A74661', '#956C89', '#8490B0', '#71B6D7', '#67CBEF'];
  const selectionColors = ['#A74661', '#956C89', '#8490B0', '#71B6D7', '#67CBEF'];
  const sketches = {
    stored: null,
    temp: null,
    debug: null,
    comp: null,
    comp2: null,
    comp_temp: null,
    selections: {
      p0: [],
      p1: [],
      pmark: []
    },
    ai: [],
    ai_overlay: []
  };
  let drawTimer = null;

  /**
  * Given a before and after at various scales and an image, return
  * suggestions for transformations to the image at those scales.
  */
  function getSuggestions(befores, afters, img, bounds, n=1) {
    console.log('fetching suggestions...');
    const data = { befores, afters, img, bounds, n };

    // console.log(befores);
    // const i = 1;
    // const container = $('#L' + (i + 1) + '_matches .selection_matches.p0');
    // container.empty();
    // const img3 = new Image();
    // img3.src = befores[i];
    // container.append(img3);
    // const container2 = $('#L' + (i + 1) + '_matches .selection_matches.p1');
    // container2.empty();
    // const img2 = new Image();
    // img2.src = afters[i];
    // container2.append(img2);


    $.get('/suggestions', data, function(result) {
      console.log(result);
      sketches.comp.clear();
      sketches.comp.noFill();
      sketches.comp2.clear();
      sketches.comp2.noFill();
      if (drawTimer) {
        clearTimeout(drawTimer);
      }

      Object.keys(result).forEach((key, i) => {
        const { locations, locationImgs, suggestions, suggestionsAligned } = result[key];
          // add suggestions at this scale to canvas
        sketches.comp2.stroke(sketches.comp.color(selectionColors[i]));

        // const drawSuggestions = (j) => {
        //   if (j >= suggestions.length - 2) {
        //     return;
        //   }
        //   drawDataURLToP(sketches.comp, suggestions[j], locations[j]);
        //   drawTimer = setTimeout(() => {
        //     const comp = sketches.comp.get();
        //     sketches.stored.image(comp, 0, 0);
        //     sketches.comp.clear();
        //     drawSuggestions(j + 1);
        //   }, 1000);
        // }
        // if (i === 1) {
        //   drawSuggestions(0);
        // }

        locations.forEach((location, j) => {
          const { x, y } = location;
          sketches.comp2.rect(x, y, selectionSizes[i], selectionSizes[i]);

          // drawDataURLToP(sketches.comp, suggestions[j], location);
          // setTimeout(() => {
          //   const comp = sketches.comp.get();
          //   sketches.stored.image(comp, 0, 0);
          //   sketches.comp.clear();
          // }, 500);
        });

        // draw suggestions to side bar
        if (locationImgs) {
          console.log('update locations')
          const container = $('#L' + (i + 1) + '_matches .selection_matches.p0');
          container.empty();
          locationImgs.forEach(dataURL => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + dataURL;
            container.append(img);
          });
        }
        if (suggestionsAligned) {
          const container = $('#L' + (i + 1) + '_matches .selection_matches.p1');
          container.empty();
          suggestionsAligned.forEach(dataURL => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + dataURL;
            container.append(img);
          });
        }
        if (suggestions) {
          const containerBase = $('#L' + (i + 1) + '_matches .selection_matches.p2.base');
          containerBase.empty();
          locationImgs.forEach(dataURL => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + dataURL;
            containerBase.append(img);
          });
          const containerOverlay = $('#L' + (i + 1) + '_matches .selection_matches.p2.overlay');
          containerOverlay.empty();
          suggestions.forEach(dataURL => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + dataURL;
            containerOverlay.append(img);
          });
        }
      })
    });
  }

  function drawDataURLToP(p, dataURL, location) {
    const raw = new Image();
    raw.src = 'data:image/jpeg;base64,' + dataURL;
    const { x, y } = location;
    raw.onload = function() {
      const img = sketches.comp_temp.createImage(raw.width, raw.height);
      img.drawingContext.drawImage(raw, 0, 0);

      // draw to graphics and threshold and make non thresh transparent
      // const pg = createGraphics(raw.width, raw.height);
      // pg.image(img, 0, 0);
      sketches.comp_temp.resizeCanvas(raw.width, raw.height);
      sketches.comp_temp.image(img, 0, 0);
      sketches.comp_temp.loadPixels();
      for (let i = 0; i < sketches.comp_temp.pixels.length; i += 4) {
        const gr = sketches.comp_temp.pixels[i];
        const o = (gr - 255) / -255;

        sketches.comp_temp.pixels[i] = 0;
        sketches.comp_temp.pixels[i + 1] = 0;
        sketches.comp_temp.pixels[i + 2] = 0;
        if (o < 0.25) {
          sketches.comp_temp.pixels[i + 3] = 0;
        } else {
          sketches.comp_temp.pixels[i + 3] = ((o * 0.3) + 0.7) * 255;
        }
      }
      sketches.comp_temp.updatePixels();
      const img_f = sketches.comp_temp.get();

      p.image(img_f, x, y);
    }
  }

  const layers_meta = [
    [
        // params for L1
        'conv1',  // layer_name
        // 71,  // output_size
        3,  // stride
        15,  // f_size
        0  // padding
    ],
    [
        // params for L2
        'conv2',  // layer_name
        // 31,  // output_size
        6,  // stride
        45,  // f_size
        0  // padding
    ],
    [
        // params for L3
        'conv3',  // layer_name
        // 15,  // output_size
        12,  // stride
        81,  // f_size
        12  // padding
    ],
    [
        // params for L4
        'conv4',  // layer_name
        // 15,  // output_size
        12,  // stride
        105,  // f_size
        24  // padding
    ],
    [
        // params for L5
        'conv5',  // layer_name
        // 15,  // output_size
        12,  // stride
        129,  // f_size
        36  // padding
    ]
  ];

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

  function getChangeSelections2(p, bounds) {
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

  const keyActions = {
    remaining: []
  };

  $(document).keypress(e => {
    console.log(e);
    if (keyActions.remaining.length > 0) {
      keyActions.remaining.forEach((layerActions, i) => {
        if (layerActions.length > 0) {
          const { location, dataURL, p } = layerActions[0];
          drawDataURLToP(p, dataURL, location);
          layerActions.splice(0, 1);
        }

        setContentsToDataURLs($('#sketch_ai' + i + '_actions'), layerActions.map(d => 'data:image/png;base64,' + d.dataURL));
      });
    }
  });

  function getActions(befores, mark, afters, imgs, bounds, n=1) {
    console.log('fetching actions...');
    const data = { befores, mark, afters, imgs, bounds, n };

    $.get('/actions', data, function(result) {
      console.log(result);
      const layers = result;

      const sketchesAIOverlay = sketches.ai_overlay;
      const sketchesAI = sketches.ai.slice(1);
      keyActions.remaining = sketchesAIOverlay.map((p, i) => {
        const layer = layers[i];
        p.clear();
        p.stroke(p.color(selectionColors[i]));
        p.strokeWeight(2);
        p.noFill();
        const { locationImgs, locations, actions } = layer;

        const actionsForLayer = [];
        setContentsToDataURLs($('#sketch_ai' + i + '_actions'), actions.map(dataURL => 'data:image/png;base64,' + dataURL));
        locations.forEach((location, j) => {
          const { x, y } = location;
          const selectionBounds = getSelectionBoundsForLayer(bounds, i);
          const w = selectionBounds[2] - selectionBounds[0];
          const h = selectionBounds[3] - selectionBounds[1];
          // p.rect(x, y, w, h);
          const nextAction = { location, dataURL: actions[j], p: sketchesAI[i] };

            lineTracer.trace(sketchesAI[i], sketchesAIOverlay[i], location, actions[j]);
          // actionsForLayer.push(nextAction);
        });
        return actionsForLayer;
      });

      const container = $('#mark_suggestions');
      container.empty();
      layers[0].marks.forEach(dataURL => {
        const img = new Image();
        img.src = 'data:image/png;base64,' + dataURL;
        container.append(img);
      });
    });
  }

  function setContentsToDataURLs(container, dataUrls) {
    container.empty();
    dataUrls.forEach(dataURL => {
      const img = new Image();
      img.src = dataURL;
      container.append(img);
    });
  }

  function onChange(bounds, lastPos) {
    console.log('Edit with bounds ' + bounds + ' and last position at ' + lastPos);
    // get previous state of change area
    const selections_p0 = getChangeSelections2(sketches.stored, bounds);
    const selections_mark = getChangeSelections2(sketches.temp, bounds);
    const mark = sketches.temp.get(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]);

    const markDataUrl = [sketches.comp_temp].map(p => {
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
    const selections_p1 = getChangeSelections2(sketches.stored, bounds);

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
    const marks = selections_mark.map((s, i) => {
      const p = sketches.selections.pmark[i];
      p.resizeCanvas(s.width, s.height);
      p.background(255);
      p.image(s, 0, 0);
      const dataURL = p.canvas.toDataURL();
      return dataURL;
    });

    // draw to display imgs
    setContentsToDataURLs($('.change_imgs .p0'), [befores[4]]);
    setContentsToDataURLs($('.change_imgs .marks'), [marks[4]]);
    setContentsToDataURLs($('.change_imgs .p1'), [afters[4]]);

    // for each ai canvas, get suggestions at the provided layers
    const imgs = sketches.ai.slice(1).map(p => p.canvas.toDataURL());

    getActions(befores, markDataUrl, afters, imgs, bounds);

    // const img = sketches.stored.canvas.toDataURL();

    // // find other areas of canvas that match p0
    // // send canvas and img selections to have activations evaluated
    // // find nearest neighbors (probably should do a 2D search tree)
    // // get similar sketch segments to p1
    // // keep massive hash tables of drawing sections (should serialize so we can quickly load)
    // // apply p1 matches to p0 matches
    // // align p1 to p0, draw using temporal agent
    // // const scale = getAppropriateScale(bounds);
    // getSuggestions(befores, afters, img, bounds);
  }

  function getAppropriateScale(bounds) {
    const change_size = Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1]);
    const size_diffs = selectionSizes.map(s => Math.abs(s - change_size))
    let min_diff = Infinity;
    let min_idx = -1;
    console.log(size_diffs);
    size_diffs.forEach((diff, i) => {
      if (diff < min_diff) {
        min_diff = diff;
        min_idx = i;
      }
    });
    const scale = min_idx;
    return scale;
  }

  function copyStoredToAI() {
    const img = sketches.stored.get();
    sketches.ai.forEach(p => p.image(img, 0, 0));
  }

  function selectAI(i) {
    const img = sketches.ai[i].get();
    sketches.stored.image(img, 0, 0);
    sketches.ai.forEach(p => p.image(img, 0, 0));
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
  new p5(getSelectionSketch(1, 0), $('#L1_matches .selection.p0')[0]);
  new p5(getSelectionSketch(1, 1), $('#L1_matches .selection.p1')[0]);
  new p5(getSelectionSketch(1, 'mark'), $('#L1_matches .selection.marks')[0]);
  new p5(getSelectionSketch(2, 0), $('#L2_matches .selection.p0')[0]);
  new p5(getSelectionSketch(2, 1), $('#L2_matches .selection.p1')[0]);
  new p5(getSelectionSketch(2, 'mark'), $('#L2_matches .selection.marks')[0]);
  new p5(getSelectionSketch(3, 0), $('#L3_matches .selection.p0')[0]);
  new p5(getSelectionSketch(3, 1), $('#L3_matches .selection.p1')[0]);
  new p5(getSelectionSketch(3, 'mark'), $('#L3_matches .selection.marks')[0]);
  new p5(getSelectionSketch(4, 0), $('#L4_matches .selection.p0')[0]);
  new p5(getSelectionSketch(4, 1), $('#L4_matches .selection.p1')[0]);
  new p5(getSelectionSketch(4, 'mark'), $('#L4_matches .selection.marks')[0]);
  new p5(getSelectionSketch(5, 0), $('#L5_matches .selection.p0')[0]);
  new p5(getSelectionSketch(5, 1), $('#L5_matches .selection.p1')[0]);
  new p5(getSelectionSketch(5, 'mark'), $('#L5_matches .selection.marks')[0]);

  function getAISketch(i) {
    return (p) => {
      sketches.ai[i] = p;
      p.setup = function setup() {
        p.pixelDensity(1);
        p.createCanvas(canvasSizeX, canvasSizeY);
        p.background(255);
        p.noLoop();
      };

      p.draw = function draw() {};
    };
  }
  new p5(getAISketch(0), $('#sketch_human')[0]);
  new p5(getAISketch(1), $('#sketch_ai' + 0)[0]);
  new p5(getAISketch(2), $('#sketch_ai' + 1)[0]);
  new p5(getAISketch(3), $('#sketch_ai' + 2)[0]);
  new p5(getAISketch(4), $('#sketch_ai' + 3)[0]);
  new p5(getAISketch(5), $('#sketch_ai' + 4)[0]);

  function getAIOverlaySketch(i) {
    return (p) => {
      sketches.ai_overlay[i] = p;
      p.setup = function setup() {
        p.pixelDensity(1);
        p.createCanvas(canvasSizeX, canvasSizeY);
        p.noLoop();
      };

      p.draw = function draw() {};
    };
  }

  new p5(getAIOverlaySketch(0, true), $('#sketch_ai0_overlay')[0]);
  new p5(getAIOverlaySketch(1, true), $('#sketch_ai1_overlay')[0]);
  new p5(getAIOverlaySketch(2, true), $('#sketch_ai2_overlay')[0]);
  new p5(getAIOverlaySketch(3, true), $('#sketch_ai3_overlay')[0]);
  new p5(getAIOverlaySketch(4, true), $('#sketch_ai4_overlay')[0]);
  $('#sketch_human').click(() => selectAI(0));
  $('#sketch_ai0_overlay').click(() => selectAI(1));
  $('#sketch_ai1_overlay').click(() => selectAI(2));
  $('#sketch_ai2_overlay').click(() => selectAI(3));
  $('#sketch_ai3_overlay').click(() => selectAI(4));
  $('#sketch_ai4_overlay').click(() => selectAI(5));

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
      //use esc to empty canvas
      if (p.keyIsPressed) {
        if (p.keyCode == p.ESCAPE) {
          p.background(255);
        }
      }
    };
  }
  new p5(sketch_stored, document.getElementById('sketch_stored'));

  function sketch_temp(p) {
    sketches.temp = p;
    let bounds = null;
    let lastPos = null;

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
          lastPos = [x, y];
        }
      }
      // If mouse is not pressed, and it was being pressed at the last draw, trigger on change and clear
      if (!p.mouseIsPressed) {
        if (bounds) {
          const pad = 10;
          bounds = [Math.max(0, bounds[0] - pad), Math.max(0, bounds[1] - pad), Math.min(p.width - 1, bounds[2] + pad), Math.min(p.height - 1, bounds[3] + pad)];
          onChange(bounds, lastPos);
        }
        bounds = null;
        lastPos = null;
      }
    };
  }
  new p5(sketch_temp, document.getElementById('sketch_temp'));

  function sketch_debug(p) {
    sketches.debug = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.noLoop();
    };

    p.draw = function draw() {
    };
  }
  new p5(sketch_debug, document.getElementById('sketch_debug'));

  function sketch_comp(p) {
    sketches.comp = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.noLoop();
    };

    p.draw = function draw() {};
  }
  new p5(sketch_comp, document.getElementById('sketch_comp'));

  function sketch_comp2(p) {
    sketches.comp2 = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.noLoop();
    };

    p.draw = function draw() {};
  }
  new p5(sketch_comp2, document.getElementById('sketch_comp2'));

  function sketch_comp_temp(p) {
    sketches.comp_temp = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.noLoop();
    };

    p.draw = function draw() {};
  }
  new p5(sketch_comp_temp, document.getElementById('sketch_comp_temp'));
}());
