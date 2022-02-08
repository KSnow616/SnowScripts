{
   function myScript(thisObj) {
      function myScript_buildUI(thisObj) {
         var myPanel = (thisObj instanceof Panel) ? thisObj : new Window("palette", "Dockable Script", undefined, { resizeable: true, closeButton: false });

         res = "group{orientation:'column',\
                         groupOne: Group{orientation:'row',\
                         AlignButton: Button{text:'Comp Align'},\
                         Info: Button{text:'?'},\
                },\
          }";

         myPanel.grp = myPanel.add(res);

         //Defaults




         myPanel.grp.groupOne.AlignButton.onClick = function () {
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

         myPanel.grp.groupOne.Info.onClick = function () {
            alert("Align and Trim a layer (or layers) to its beneath layer. v.0.1.0 By KSnow.")
         }
         myPanel.grp.groupOne.Info.size = [25, 25];
         myPanel.layout.layout(true);

         return myPanel;
      }




      var myScriptPal = myScript_buildUI(thisObj);

      if (myScriptPal != null && myScriptPal instanceof Window) {
         myScriptPal.center();
         myScriptPal.show();
      }

   }
   myScript(this);
}



