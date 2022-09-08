{
   function myScript(thisObj) {
      function myScript_buildUI(thisObj) {
         var myPanel = (thisObj instanceof Panel) ? thisObj : new Window("palette", "Dockable Script", undefined, { resizeable: true, closeButton: false }); // init the panel

         res = "group{orientation:'column',\
                         groupOne: Group{orientation:'row',\
                         AlignButton: Button{text:'Layer Align'},\
                         Info: Button{text:'?'},\
                },\
          }"; //Configurate UI

         myPanel.grp = myPanel.add(res);

         myPanel.grp.groupOne.AlignButton.onClick = function () { // do the function when clicked (core part)
            try {
               var comp = app.project.activeItem;
               var slLayers = comp.selectedLayers;
               var maxindex = 0;
               for (i = 0; i < slLayers.length; i++) {
                  if (slLayers[i].index > maxindex) {
                     maxindex = slLayers[i].index;
                  }
               }

               for (i = 0; i < slLayers.length; i++) {
                  slLayers[i].inPoint = comp.layer(maxindex + 1).inPoint
                  slLayers[i].outPoint = comp.layer(maxindex + 1).outPoint
               }
            } catch (error) {
               alert("Please select a layer.")
            }

         }

         myPanel.grp.groupOne.Info.onClick = function () { // Info button
            alert("Align and Trim a layer (or layers) to its beneath layer. v.0.1.0 By KSnow.")
         }
         myPanel.grp.groupOne.Info.size = [25, 25]; // Control the size of the info buton
         myPanel.layout.layout(true);

         return myPanel;
      } // end of myScript_buildUI()




      var myScriptPal = myScript_buildUI(thisObj);

      if (myScriptPal != null && myScriptPal instanceof Window) {
         myScriptPal.center();
         myScriptPal.show();
      }

   }
   myScript(this);
}



