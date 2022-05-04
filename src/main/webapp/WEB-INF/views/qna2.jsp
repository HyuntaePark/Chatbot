<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script>

function askHelp(){
   var time = new Date();
   time =  time.getHours().toString().padStart(2,'0') + ":" + time.getMinutes().toString().padStart(2,'0');

   $("#result").append(
         "<div class=" + '"message sol">' 
         + '<div class="messageText" data-time="' 
         + time +'">'
         + "안녕하세요. 민원 상담 서비스 입니다. 무엇을 도와드릴까요?" + "</div></div>"
         );
}

$(function(){
   $("input[type=button]").click(function(){
      req_url = "http://localhost:5000/ai_bot"
      var form = $("form")[0];
      var form_data = new FormData(form);
      var time = new Date();
      time =  time.getHours().toString().padStart(2,'0') + ":" + time.getMinutes().toString().padStart(2,'0');
      
      $.ajax({
         url: req_url,
         async : true,
         type : "POST",
         data: form_data,
         processData : false,
         contentType : false,
         success :  function(data){
               console.log(data)
               question = $("input[name=question]").val();
               $("#result").append(
               '<div class="message sag mtLine"><div class="messageText" data-time="'+time+'">'
               +question+ '</div></div>'
               +"<div class=" 
               + '"message sol">' + '<div class="messageText" data-time="'+time+'">'+
               data + "</div></div>"
               );
               $("input[name=question]").val("")
            },

         error: function(e){
            alert(e);
         }
      })
      question = $("input")
   })
})
</script>
</head>

<style type="text/css">
body {
  background: #222;
}

* {
  outline: 0;
}

.time {
  text-align: center;
  margin-bottom: 10px;
}
.time span {
  background-color: #000000;
  display: inline-block;
  border-radius: 3px;
  text-align: center;
  padding: 2px 20px;
  color: #fff;
  opacity: 0.3;
}
.message {
  margin-bottom: 10px;
}
.message .messageText {
  text-align: left;
  color: #FFFFFF;
}
.message.sol {
  text-align: left;
}
.message.sag {
  text-align: right;
}
.message .resim {
  background: #FF0044 none no-repeat center;
  vertical-align: text-top;
  background-size: cover;
  display: inline-block;
  position: relative;
  color: #FFFFFF;
  height: 40px;
  width: 40px;
}
.message .messageText {
  background-color: #FF0044;
  vertical-align: text-top;
  display: inline-block;
  position: relative;
  line-height: 20px;
  max-width: 165px;
  color: #FFFFFF;
  padding: 10px;
}
.message.left .userPortrait,
.message.sag .messageText {
  border-radius: 5px 0 0 5px;
}
.message.sag .userPortrait,
.message.sol .messageText {
  border-radius: 0 5px 5px 5px;
}
.message.mtLine.sol .messageText {
  border-radius: 0 5px 5px 0;
}
.message.mtLine.sag .messageText {
  border-radius: 5px 5px 0 5px;
}
.message .messageText:before {
  border-color: transparent #FF0044;
  border-style: solid;
  position: absolute;
  border-width: 0;
  display: block;
  content: "";
  z-index: 1;
}
.message.sol .messageText:before {
  border-width: 0 10px 10px 0;
  left: -10px;
  top: 0;
}
.message.sag .messageText:before {
  border-width: 10px 0 0 10px;
  right: -10px;
  top: 30px;
}
.message .messageText:after {
  content: attr(data-time);
  color: rgba(255,255,255,0.5);
  position: absolute;
  line-height: 20px;
  display: block;
  bottom: 2px;
  z-index: 1;
}
.message.sol .messageText:after {
  margin-left: 5px;
  left: 100%;
}
.message.sag .messageText:after {
  margin-right: 5px;
  right: 100%;
}

#input1{
   width: 80%;
   height: 30px;
   border: 3px solid #FF0044;
   border-radius: 5px;
}

#input2{
   width: 100px;
   height: 40px;
   border: 3px solid #FF0044;
   border-radius: 5px;
}

}

</style>

<body onload="askHelp()">
   <div class="time"><span>민원 상담 챗봇 서비스</span></div>
    <div id="result" ></div>
    <form action="#" method="post">
          <input id="input1" type="text" name = "question" size="30" placeholder="질문을 적어주세요!">
          <input id ="input2" type="button" value="입력">
    </form>
    

</html>