function showFileName(elementId, inputElement) {
    const fileName = "Resim/Resimler seçildi!"
    const fileNameDisplay = document.getElementById(elementId);
    fileNameDisplay.textContent = fileName;
}

function filter(){

  var emojiSelect= document.getElementById("emojiSelectBox")

  var filterSelect = document.getElementById("filterSelect");
  var filterName = filterSelect.value;

  if(filterName == "emojiFace"){ 

    var emojiValue = ["smileyFace","coolFace","sadFace","angelFace"];
    var emojiDec = ["128512","128526","128532","128519"]

    var selectList = document.createElement("select");
    selectList.id = "emojiSelect";
    selectList.name ="emojiSelect";
    selectList.style.margin = "0px 0px 10px 0px"
    emojiSelect.appendChild(selectList);

    for (var i = 0; i < emojiValue.length; i++) {
      var option = document.createElement("option");
      option.value = emojiValue[i];
      var dec = emojiDec[i];
      option.text =  String.fromCodePoint(dec)
      selectList.appendChild(option);
    }
  }

  var element = document.getElementById("emojiSelect")
  if(filterName !== "emojiFace" && element != null ){
    document.getElementById("emojiSelect").remove(); 
  }
}

