
var button=document.getElementById("submit")
var inp=document.getElementById("input")
var pos=document.getElementById("pos")
var neg=document.getElementById("neg")
var  img=document.getElementById("img")
button.onclick=function(){

    var data = new FormData();
    data.append("query", inp.value);
    
    var xhr = new XMLHttpRequest();
    
    
    xhr.addEventListener("readystatechange", function () {
      if (this.readyState === 4) {
        console.log(this.responseText);
        var response=JSON.parse(this.responseText);
        
        pos.innerHTML="positive:  "+response.positive;
        neg.innerHTML="negative:  "+response.negative;
        img.src="img_"+response.id+".png";
      }
    });
    
    xhr.open("POST", "http://127.0.0.1:8000/predict");
    xhr.setRequestHeader("Cache-Control", "no-cache");
    
    xhr.send(data);


}