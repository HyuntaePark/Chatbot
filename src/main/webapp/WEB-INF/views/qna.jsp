<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script>
$(function(){
	$("input[type=button]").click(function(){
		req_url = "http://localhost:5000/ai_bot"
		var form = $("form")[0];
		var form_data = new FormData(form);
		$.ajax({
			url: req_url,
			async : true,
			type : "POST",
			data: form_data,
			processData : false,
			contentType : false,
			success : function(data){
				console.log(data)
				question = $("input[name=question]").val();
				$("#result").append("<p>Q:" + question + "<br>A:" + data + "</p>");
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
<body>
<h1>질문을 입력하세요.</h1>
<form action="#" method="post">
Question: <input type="text" name = "question" size="50">
<input type = "button" value = "물어보기">
</form>
<div id="result"></div>
</body>
</html>