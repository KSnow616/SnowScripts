{
    function myScript(thisObj) {
       function myScript_buildUI(thisObj) {
          var myPanel = (thisObj instanceof Panel) ? thisObj : new Window("palette", "Dockable Script", undefined, { resizeable: true, closeButton: false }); // init the panel
 
          res = "group{orientation:'column',\
                          groupOne: Group{orientation:'row',\
                          DistressButton: Button{text:'Distress'},\
                 },\
           }"; //Configurate UI
 
          myPanel.grp = myPanel.add(res);
 
          myPanel.grp.groupOne.DistressButton.onClick = function () { // do the function when clicked (core part)
             try {
                var file = new File;
                var file = new File("C:\\Users\\snow\\Documents\\distress_button.txt"); //change this for your txt
                var textArray = [];
                var currentLine;
                var lineNum = 0;
                file.open("r");

                while(!file.eof){
                    currentLine = file.readln();
                    textArray.push(currentLine);
                    lineNum++;
                }
                
                file.close();

                alert(textArray[Math.floor(Math.random()*lineNum)]);
             } catch (error) {
                alert("404 (File not found.)")
             }
 
          }
 
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
 
 
 
 